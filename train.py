#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#
# Modified by Byoungkwon Yoon and contributors, 2026.
# Changes include LiDAR depth supervision, geometric variance-based
# densification, and training code refactoring.
#

import os
import uuid
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from random import randint

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, update_room
from scene import Scene, TriangleModel
from triangle_renderer import render
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


@dataclass
class TrainingLosses:
    """Loss terms, including their configured weights, for one training view."""

    pixel: torch.Tensor
    image: torch.Tensor
    weight: torch.Tensor | int
    normal: torch.Tensor
    depth: torch.Tensor
    distortion: torch.Tensor
    total: torch.Tensor


@dataclass
class TrainingBuffers:
    """Supervision and errors shared by variance updates and debug plots."""

    gt_normal: torch.Tensor
    gt_depth: torch.Tensor
    valid_depth: torch.Tensor
    rendered_normal: torch.Tensor
    normal_error: torch.Tensor
    depth_error: torch.Tensor


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    checkpoint,
    debug_from,
    use_wandb=False,
    wandb_project="triangle-gs",
    wandb_name=None,
    lpips_fn=None,
):
    """Optimize the scene, update its topology, then prune and save it.

    W&B arguments remain accepted for compatibility; logging uses TensorBoard.
    Callers may supply an LPIPS evaluator; main() creates it before RNG seeding.
    """
    if lpips_fn is None:
        lpips_fn = lpips.LPIPS(net="vgg").to(device="cuda")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(
        opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init
    )
    triangles.add_percentage = opt.add_percentage

    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    prune_threshold = opt.prune_triangles_threshold
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space

    for iteration in range(first_iter, opt.iterations + 1):
        if iteration == opt.start_upsampling:
            triangles.scaling = opt.upscaling_factor

        iter_start.record()
        triangles.update_learning_rate(iteration)
        if iteration < opt.sigma_start:
            current_sigma = opt.set_sigma
        else:
            progress = (iteration - opt.sigma_start) / (
                opt.sigma_until - opt.sigma_start
            )
            progress = min(progress, 1.0)
            current_sigma = opt.set_sigma - (opt.set_sigma - 0.0001) * progress
        triangles.set_sigma(current_sigma)

        # Increase the spherical harmonic degree every 1,000 iterations.
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand(3, device="cuda") if opt.random_background else background
        if (
            not viewpoint_stack
            or len(scene.getTrainCameras()) + iteration == opt.iterations
        ):
            viewpoint_stack = scene.getTrainCameras().copy()
            if len(scene.getTrainCameras()) + iteration == opt.iterations:
                print(iteration)
                # Recompute importance over the last camera sweep.
                triangles.importance_score = torch.zeros(
                    triangles._triangle_indices.shape[0],
                    dtype=torch.float,
                    device="cuda",
                )
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        render_pkg = render(viewpoint_cam, triangles, pipe, bg)

        losses, buffers = compute_training_losses(
            viewpoint_cam, render_pkg, triangles, opt, iteration
        )
        losses.total.backward()
        update_triangle_variances(triangles, render_pkg, buffers, opt, iteration)

        if iteration % 100 == 0:
            save_debug_visualization(dataset.model_path, iteration, render_pkg, buffers)
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * losses.total.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.5f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                losses.pixel,
                losses.total,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                use_wandb=use_wandb,
                lpips_fn=lpips_fn,
            )
            if iteration % 500 == 0:
                prune_threshold = update_topology(
                    triangles, opt, iteration, prune_threshold
                )

            # The final iteration is evaluated and saved without an optimizer step.
            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)

    # Cleanup uses the last training background, including random backgrounds.
    # prune_final_triangles(scene, pipe, bg)
    print("Training is done")
    scene.save(iteration)


def compute_training_losses(viewpoint, render_pkg, triangles, opt, iteration):
    """Compute the objective and retain buffers used by geometry and plotting."""
    image = render_pkg["render"]
    gt_image = viewpoint.original_image.cuda()
    gt_normal = F.interpolate(
        viewpoint.normal_map.cuda().unsqueeze(0),
        size=(gt_image.shape[1], gt_image.shape[2]),
        mode="area",
    ).squeeze(0)
    pixel_loss = l1_loss(image, gt_image)

    # Retain the largest projected size and blending contribution across views.
    image_size = render_pkg["scaling"].detach()
    mask = image_size > triangles.image_size
    triangles.image_size[mask] = image_size[mask]
    importance_score = render_pkg["max_blending"].detach()
    mask = importance_score > triangles.importance_score
    triangles.importance_score[mask] = importance_score[mask]

    image_loss = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (
        1.0 - ssim(image, gt_image)
    )
    rendered_normal = F.normalize(render_pkg["rend_normal"], dim=0)
    lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
    normal_error = 1 - (rendered_normal * gt_normal).sum(dim=0)
    normal_loss = lambda_normal * normal_error.mean()

    if iteration < opt.start_opacity_floor:
        weight_loss = (
            triangles.get_vertex_weight[triangles._triangle_indices].mean()
            * opt.lambda_weight
        )
    else:
        weight_loss = 0

    gt_depth = viewpoint.depth_map.cuda()
    valid_depth = (gt_depth < 15) & (gt_depth > 0)
    depth_error = torch.abs(render_pkg["surf_depth"] - gt_depth)
    depth_error[~valid_depth] = 0
    # Preserve the original mean, including NaN when a view has no valid depth.
    depth_loss = opt.lambda_depth * depth_error[valid_depth].mean()
    distortion_loss = opt.lambda_dist * render_pkg["rend_dist"].mean()
    total_loss = image_loss + weight_loss + normal_loss + depth_loss + distortion_loss

    losses = TrainingLosses(
        pixel=pixel_loss,
        image=image_loss,
        weight=weight_loss,
        normal=normal_loss,
        depth=depth_loss,
        distortion=distortion_loss,
        total=total_loss,
    )
    buffers = TrainingBuffers(
        gt_normal=gt_normal,
        gt_depth=gt_depth,
        valid_depth=valid_depth,
        rendered_normal=rendered_normal,
        normal_error=normal_error,
        depth_error=depth_error,
    )
    return losses, buffers


def update_triangle_variances(triangles, render_pkg, buffers, opt, iteration):
    """Accumulate per-triangle error variance and smooth it before upsampling."""
    with torch.no_grad():
        flat_ids = render_pkg["rend_ids"].view(-1).long()
        flat_normal_error = buffers.normal_error.view(-1)
        # The original NumPy round trip converted depth errors to float32.
        flat_depth_error = buffers.depth_error.detach().float().flatten()
        valid_mask = (flat_ids >= 0) & buffers.valid_depth.view(-1)
        valid_ids = flat_ids[valid_mask]
        normal_error = flat_normal_error[valid_mask]
        depth_error = flat_depth_error[valid_mask]
        num_triangles = triangles._triangle_indices.shape[0]

        pixel_count = torch.zeros(num_triangles, device="cuda")
        pixel_count.index_add_(
            0, valid_ids, torch.ones_like(valid_ids, dtype=torch.float)
        )
        sum_normal = torch.zeros(num_triangles, device="cuda")
        sum_normal.index_add_(0, valid_ids, normal_error)
        sq_sum_normal = torch.zeros(num_triangles, device="cuda")
        sq_sum_normal.index_add_(0, valid_ids, normal_error**2)
        sum_depth = torch.zeros(num_triangles, device="cuda")
        sum_depth.index_add_(0, valid_ids, depth_error)
        sq_sum_depth = torch.zeros(num_triangles, device="cuda")
        sq_sum_depth.index_add_(0, valid_ids, depth_error**2)

        # Var(X) = E[X²] - E[X]², using valid LiDAR pixels (including zero error).
        safe_count = torch.clamp(pixel_count, min=1.0)
        normal_variance = sq_sum_normal / safe_count - (sum_normal / safe_count) ** 2
        depth_variance = sq_sum_depth / safe_count - (sum_depth / safe_count) ** 2
        normal_variance[pixel_count <= 1] = 0.0
        depth_variance[pixel_count <= 1] = 0.0

    if iteration < opt.start_upsampling:
        alpha = 0.2
        new_normal_variance = (
            1 - alpha
        ) * triangles.normal_var_probs.detach() + alpha * normal_variance
        new_depth_variance = (
            1 - alpha
        ) * triangles.depth_var_probs.detach() + alpha * depth_variance
        triangles.update_normal_var_probs(new_normal_variance)
        triangles.update_depth_var_probs(new_depth_variance)


def save_debug_visualization(model_path, iteration, render_pkg, buffers):
    """Save the six-panel normal, depth, error, and RGB training diagnostic."""
    image = render_pkg["render"]
    rend_normal = buffers.rendered_normal
    gt_normal = buffers.gt_normal
    normal_error = buffers.normal_error
    rend_depth = render_pkg["surf_depth"]
    gt_depth = buffers.gt_depth
    f, axarr = plt.subplots(3, 2, figsize=(18, 12))
    rend_normal_img = rend_normal.detach().cpu().float().numpy()
    rend_normal_img = np.moveaxis(rend_normal_img, 0, -1)
    rend_normal_img = 0.5 * (rend_normal_img + 1.0)
    rend_normal_img = rend_normal_img * 255
    gt_normal_img = gt_normal.cpu().float().numpy()
    gt_normal_img = np.moveaxis(gt_normal_img, 0, -1)
    gt_normal_img = 0.5 * (gt_normal_img + 1.0)
    gt_normal_img = gt_normal_img * 255
    normal_error_img = normal_error.detach().cpu().float().numpy()
    normal_error_img = np.moveaxis(normal_error_img, 0, -1)
    normal_error_img = normal_error_img * 128
    axarr[0, 0].imshow(rend_normal_img.astype(np.uint8))
    axarr[0, 0].set_title("rendered normal")
    axarr[0, 1].imshow(gt_normal_img.astype(np.uint8))
    axarr[0, 1].set_title("gt normal")

    rend_depth_img = rend_depth.detach().cpu().float().numpy()
    rend_depth_img = np.moveaxis(rend_depth_img, 0, -1)
    im_depth = axarr[1, 0].imshow(rend_depth_img, cmap="viridis")
    f.colorbar(im_depth, ax=axarr[1, 0])
    axarr[1, 0].set_title("rendered depth")

    gt_depth_img = gt_depth.detach().cpu().float().numpy()
    gt_depth_img = np.moveaxis(gt_depth_img, 0, -1)
    im_gt_depth = axarr[1, 1].imshow(gt_depth_img, cmap="viridis")
    f.colorbar(im_gt_depth, ax=axarr[1, 1])
    axarr[1, 1].set_title("GT depth")

    axarr[2, 0].imshow(normal_error_img.astype(np.uint8), cmap="gray")
    axarr[2, 0].set_title("normal error")
    axarr[2, 1].imshow(
        np.clip(np.moveaxis(image.detach().cpu().float().numpy(), 0, -1), 0, 1)
    )
    axarr[2, 1].set_title("iteration: " + str(iteration))
    os.makedirs(os.path.join(model_path, "debug_img"), exist_ok=True)
    plt.savefig(os.path.join(model_path, "debug_img", str(iteration) + ".png"), dpi=300)
    plt.close()


def update_topology(triangles, opt, iteration, prune_threshold):
    """Prune, densify, and raise the opacity floor; return the next threshold.

    Called every 500 iterations under no_grad, before the optimizer step.
    """
    print(torch.min(triangles.importance_score))
    triangle_weights = triangles.opacity_activation(
        triangles.vertex_weight[triangles._triangle_indices]
    )
    min_weights = triangle_weights.min(dim=1).values
    mask_opacity = (min_weights <= prune_threshold).squeeze()
    mask_importance = (
        triangles.importance_score <= opt.prune_importance_threshold
    ).squeeze()
    mask_size = (triangles.image_size > opt.prune_size).squeeze()
    keep_mask = ~(mask_opacity | mask_importance | mask_size)

    if iteration > opt.start_pruning:
        print("pruning...")
        triangles.prune_triangles(keep_mask)

    # Keep vertices referenced by triangles or meeting the opacity threshold.
    used_vertex_mask = torch.zeros(
        triangles.vertices.shape[0], dtype=torch.bool, device=triangles.vertices.device
    )
    if triangles._triangle_indices.numel() > 0:
        used_vertex_mask[triangles._triangle_indices.flatten()] = True
    weight_mask = triangles.get_vertex_weight.squeeze() >= prune_threshold
    triangles._prune_vertices(weight_mask | used_vertex_mask)

    needs_densification = (
        iteration < opt.densify_until_iter
        and iteration % opt.densification_interval == 0
        and iteration > opt.densify_from_iter
    )
    if needs_densification:
        print("densifing...")
        probs_opacity = iteration < opt.start_opacity_floor or iteration % 1000 == 0
        triangles.add_new_gs_VAR(
            iteration,
            cap_max=opt.max_points,
            splitt_large_triangles=opt.splitt_large_triangles,
            probs_opacity=probs_opacity,
        )

    if iteration > opt.start_opacity_floor:
        final_opacity = 0.9999
        progress = min(
            1.0,
            max(
                0.0,
                (iteration - opt.start_opacity_floor)
                / max(1, opt.final_opacity_iter - opt.start_opacity_floor),
            ),
        )
        current_opacity = opt.set_weight + (final_opacity - opt.set_weight) * progress
        triangles.update_min_weight(min(current_opacity, final_opacity))
        prune_threshold += 0.01
    return prune_threshold

def prepare_output_and_logger(args):
    """Create the output directory, save model arguments, and open TensorBoard."""
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    pixel_loss,
    loss,
    loss_fn,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    use_wandb=False,
    lpips_fn=None,
):
    """Log losses each step and evaluate every 1,000 steps using the supplied LPIPS model."""
    if tb_writer:
        tb_writer.add_scalar(
            "train_loss_patches/pixel_loss", pixel_loss.item(), iteration
        )
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # testing_iterations controls ground-truth image logging, not evaluation cadence.
    # use_wandb is accepted for compatibility but has no logging implementation.
    if iteration % 1000 == 0:
        if lpips_fn is None:
            raise ValueError("Evaluation requires an LPIPS evaluator via lpips_fn.")
        if iteration in [5000, 13000]:
            scene.save(iteration)
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = renderFunc(viewpoint, scene.triangles, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    torch.cuda.synchronize()

                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )

                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    cur_psnr = psnr(image, gt_image).mean().double()
                    psnr_test += cur_psnr

                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                pixel_loss_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                lpips_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                        iteration,
                        config["name"],
                        pixel_loss_test,
                        psnr_test,
                        ssim_test,
                        lpips_test,
                    )
                )

                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss",
                        pixel_loss_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        torch.cuda.empty_cache()


def main(argv=None):
    """Parse training options and run the experiment."""
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    # Evaluation uses a fixed cadence; this list controls ground-truth image logging.
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    # Compatibility options: save/checkpoint lists are currently not consumed.
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[13_000, 17_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--room", action="store_true", default=False)
    # Compatibility options: W&B logging is currently not implemented.
    parser.add_argument(
        "--use_wandb", action="store_true", default=False, help="Enable wandb logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="triangle-gs", help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Wandb run name (defaults to model_path basename)",
    )

    args = parser.parse_args(argv)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Preserve model initialization before safe_state() resets the RNG.
    lpips_fn = lpips.LPIPS(net="vgg").to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    lps = lp.extract(args)
    ops = op.extract(args)
    pps = pp.extract(args)

    if args.room:
        ops = update_room(ops)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lps,
        ops,
        pps,
        args.test_iterations,
        args.start_checkpoint,
        args.debug_from,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        lpips_fn=lpips_fn,
    )

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()

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

# Modified by Byoungkwon Yoon and contributors, 2025-2026.
# Changes add depth correlation and directional point-cloud loss functions.

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics.functional.regression import pearson_corrcoef
from math import exp
from pytorch3d.ops import knn_points


def triangle_probability_loss(vertices, _triangle_indices, _closest_vertices, theta=10.0):
    """
    Computes the probability loss for triangles to maximize their separation from nearby points.
    
    For each triangle:
    1. Computes circumcenter and circumradius
    2. Finds closest point among K neighbors
    3. Calculates signed distance (closest_distance - circumradius)
    4. Computes probability = sigmoid(signed_distance * theta)
    
    Returns the negative mean probability (loss to minimize)
    """
    # Get triangle vertices [T, 3, 3]
    tri_pts = vertices[_triangle_indices]
    
    # Compute circumcenters and circumradii
    A, B, C = tri_pts.unbind(dim=1)
    AB = B - A
    AC = C - A
    n = torch.cross(AB, AC, dim=1)
    
    # Vector magnitudes
    AB2 = (AB * AB).sum(dim=1, keepdim=True)
    AC2 = (AC * AC).sum(dim=1, keepdim=True)
    
    # Denominator (2 * |n|^2)
    den = 2.0 * (n * n).sum(dim=1, keepdim=True)
    
    # Identify degenerate triangles (den near zero)
    eps = 1e-12
    degenerate = (den.abs() < eps).squeeze(1)
    
    # Compute circumcenters (use centroid for degenerate triangles)
    term1 = AB2 * torch.cross(AC, n, dim=1)
    term2 = AC2 * torch.cross(n, AB, dim=1)
    circumcenters = A + (term1 + term2) / den
    
    # For degenerate triangles, use centroid instead
    centroid = tri_pts.mean(dim=1)
    circumcenters = torch.where(degenerate.unsqueeze(1), centroid, circumcenters)
    
    # Compute circumradii (distance from circumcenter to vertices)
    dists_to_vertices = torch.norm(tri_pts - circumcenters.unsqueeze(1), dim=2)
    circumradii = dists_to_vertices.max(dim=1).values
    
    # Get closest points for each triangle [T, K, 3]
    closest_points = vertices[_closest_vertices]
    
    # Compute distances to closest points
    dists_to_neighbors = torch.norm(
        closest_points - circumcenters.unsqueeze(1),
        dim=2
    )
    
    # Find minimum distance for each triangle
    min_dists, _ = torch.min(dists_to_neighbors, dim=1)
    
    # Signed distance = (closest distance) - circumradius
    signed_dist = min_dists - circumradii
    
    # For degenerate triangles, set signed_dist to large negative value
    #signed_dist = torch.where(degenerate, -1e6 * torch.ones_like(signed_dist), signed_dist)
    
    # Compute probability using sigmoid
    probability = torch.sigmoid(-theta * signed_dist)
    
    # Loss is negative mean probability (to maximize probability)
    return torch.mean(probability)


def u_shaped_opacity_loss(x, center=0.1, width=0.03):
    # Normalized distance to center (e.g., 0.1)
    penalty = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
    return penalty.mean()

def binarization_loss(x, eps=1e-6):
    x = torch.clamp(x, eps, 1 - eps)  # avoid log(0)
    return -x * torch.log(x) - (1 - x) * torch.log(1 - x)

def equilateral_regularizer(triangles):

    nan_mask = torch.isnan(triangles).any(dim=(1, 2))
    if nan_mask.any():
        print("NaN detected in triangle(s):")

    v0 = triangles[:, 1, :] - triangles[:, 0, :]
    v1 = triangles[:, 2, :] - triangles[:, 0, :]
    cross = torch.cross(v0, v1, dim=1)
    area = 0.5 * torch.norm(cross, dim=1)

    return area


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def lp_loss(pred, target, p=0.7, eps=1e-6):
    """
    Computes Lp loss with 0 < p < 1.
    Args:
        pred: (N, C, H, W) predicted image
        target: (N, C, H, W) groundtruth image
        p: norm degree < 1
        eps: small constant for numerical stability
    """
    diff = torch.abs(pred - target) + eps
    loss = torch.pow(diff, p).mean()
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def cauchy_loss(x, y, reduction="mean"):
    loss_map = torch.log1p(torch.square(x - y))
    if reduction == "sum":
        return loss_map.sum()
    if reduction == "mean":
        return loss_map.mean()
    raise NotImplementedError

def pearson_loss(render, estimate, mask=None, invert_estimate=True):
    """
    Calculate loss using the pearson correlation coefficient.
    We want the pearson correlation to approach 1.
    """
    if mask is not None:
        render_masked = render[mask]
        estimate_masked = estimate[mask]
    else:
        render_masked = render
        estimate_masked = estimate
    if invert_estimate:
        loss = min(
            (1 - pearson_corrcoef(-estimate_masked, render_masked)).mean(),
            (1 - pearson_corrcoef(1 / (estimate_masked + 200.0), render_masked)).mean(),
        )
    else:
        loss = (1 - pearson_corrcoef(estimate_masked, render_masked)).mean()
    return loss

def directional_knn_loss(triangle_model, vis, gt_points, K=20, padding=0.1):
    """
    triangle_model: The TriangleModel object
    vis: visibility_filter mask [N_triangles]
    gt_points: (M, 3) tensor of LiDAR points
    K: Top-K candidates for directional alignment
    padding: Meters to expand the bounding box for cropping GT points
    """
    # 1. Extract Geometry
    vertices = triangle_model.get_vertices
    indices = triangle_model._triangle_indices[vis]
    
    # Get unique vertex indices that are actually visible to current camera
    visible_v_idx = torch.unique(indices.flatten())
    local_vertices = vertices[visible_v_idx]
    
    if local_vertices.shape[0] == 0:
        return torch.tensor(0.0, device=vertices.device)

    with torch.no_grad():
        # 2. Crop GT Points using Bounding Box of local_vertices
        v_min = local_vertices.min(dim=0).values - padding
        v_max = local_vertices.max(dim=0).values + padding
        
        # Fast boolean filtering for GT points within the box
        gt_mask = (gt_points[:, 0] >= v_min[0]) & (gt_points[:, 0] <= v_max[0]) & \
                  (gt_points[:, 1] >= v_min[1]) & (gt_points[:, 1] <= v_max[1]) & \
                  (gt_points[:, 2] >= v_min[2]) & (gt_points[:, 2] <= v_max[2])
        
        cropped_gt = gt_points[gt_mask]
        
        # Fallback if no GT points are in the box (e.g., initialization)
        if cropped_gt.shape[0] < K:
            cropped_gt = gt_points 

        # 3. Calculate Triangle Centers (only for visible triangles)
        v0, v1, v2 = vertices[indices[:, 0]], vertices[indices[:, 1]], vertices[indices[:, 2]]
        centers = (v0 + v1 + v2) / 3.0 # (F_visible, 3)

        # 4. Map centers back to unique visible vertices
        vertex_centers = torch.zeros_like(local_vertices)
        counts = torch.zeros((local_vertices.shape[0], 1), device=vertices.device)
        
        # We need to remap the global indices to local (0 to N_visible)
        # using a search-sorted or a temporary mapping array
        max_idx = visible_v_idx.max() + 1
        remap = torch.zeros(max_idx.item(), dtype=torch.long, device=vertices.device)
        remap[visible_v_idx] = torch.arange(visible_v_idx.shape[0], device=vertices.device)
        
        local_indices = remap[indices] # (F_visible, 3)

        vertex_centers.index_add_(0, local_indices[:, 0], centers)
        vertex_centers.index_add_(0, local_indices[:, 1], centers)
        vertex_centers.index_add_(0, local_indices[:, 2], centers)
        
        ones = torch.ones((local_indices.shape[0], 3), device=vertices.device)
        counts.index_add_(0, local_indices.flatten(), ones.flatten().unsqueeze(-1))
        vertex_centers /= (counts + 1e-6)

        # 5. Define Search Direction
        direction = local_vertices - vertex_centers
        direction = torch.nn.functional.normalize(direction, dim=-1)

        # 6. Top-K KNN Search on Cropped GT
        v_batch = local_vertices.unsqueeze(0)
        gt_batch = cropped_gt.unsqueeze(0)
        knn = knn_points(v_batch, gt_batch, K=K)
        
        neighbor_points = cropped_gt[knn.idx.squeeze(0)] 

        # 7. Select Best Directional Match
        v_to_p = neighbor_points - local_vertices.unsqueeze(1)
        alignment = torch.sum(v_to_p * direction.unsqueeze(1), dim=-1)
        
        ortho_dist_sq = torch.sum(v_to_p**2, dim=-1) - (alignment**2)
        score = ortho_dist_sq + 0.1 * torch.sum(v_to_p**2, dim=-1) 
        best_idx = torch.argmin(score, dim=1)
        
        target_points = neighbor_points[torch.arange(len(local_vertices)), best_idx].detach()

    # 8. Final Differentiable Loss (only on visible vertices)
    return torch.mean(torch.sum((local_vertices - target_points)**2, dim=-1))
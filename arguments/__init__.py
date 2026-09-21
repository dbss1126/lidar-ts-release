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
# Changes add and tune LiDAR supervision and coarse-to-fine training parameters.

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 1.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.lambda_dssim = 0.2

        self.densification_interval = 500  # default: 500

        self.densify_from_iter = 500   # default: 500
        self.densify_until_iter = 20000 # default: 13000

        self.random_background = False
        
        self.feature_lr = 0.009 # 0.0025 / 0.007
        self.max_points = 5000000 # default: 5000000

        # Opacity & weight
        self.set_weight = 0.5   # default:0.28 / 0.2
        self.weight_lr =  0.03  # default:0.03 / 0.005
        self.lambda_weight = 1.9e-06

        # Normal loss
        self.iteration_mesh = 2000  # default: 5000
        # self.lambda_normals = .3  # default: 0.05
        self.lambda_normals = 0.0001 # default: 0.05

        # self.add_percentage = 1.3  # default: 1.23
        self.add_percentage = 1.1  # default: 1.2

        # Depth loss
        # self.lambda_depth = 0.6
        self.lambda_depth = 0.0015  # default: 0.001

        # Depth distirtion loss
        self.lambda_dist = 10

        # PARAMETER FIRST STAGE
        self.set_sigma =  1.0   # default: 1.0 

        # Add new triangles or vertices
        self.intervall_add_triangles = 500

        # Prune triangles and vertices
        self.prune_triangles_threshold = 0.1 ## default: 0.235
        self.prune_importance_threshold = 0.0

        # PARAMETER SECOND STAGE
        self.lr_triangles_points_init = 0.001  ## default: 0.0015

        self.start_opacity_floor = 8000 

        self.start_pruning = 4000
        self.sigma_until = 30000    ## default: 30000
        self.final_opacity_iter = 24000

        self.sigma_start = 30000  ## default: 30000

        self.splitt_large_triangles =  100 #default: 100 / 1000 / split top N largest triangles
        self.start_upsampling = 20000
        self.upscaling_factor = 1

        self.size_probs_zero = 7.5e-05
        self.size_probs_zero_image_space = 10.0

        self.prune_size = 1400

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def update_room(params):
    print("+++++++++++++++++ start room +++++++++++++++++++++")
    params.iterations = 13_000
    params.position_lr_delay_mult = 0.001
    params.position_lr_max_steps = 30_000
    params.lambda_dssim = 0.2

    params.densification_interval = 500

    params.densify_from_iter = 500
    params.densify_until_iter = 12999

    params.random_background = False
        
    params.feature_lr = 0.0025 
    params.max_points = 6000000

        # Opacity & weight
    params.set_weight = 0.28
    params.weight_lr =  0.03
    params.lambda_weight = 1.9e-06

        # Normal loss
    params.iteration_mesh = 1000
    params.lambda_normals = 0.05
    params.lambda_depth = 0.001
    params.lambda_dist = 1

    params.add_percentage = 1.05

        # PARAMETER FIRST STAGE
    params.set_sigma = 1.0
    params.set_weight = 0.28

        # Add new triangles or vertices
    params.intervall_add_triangles = 500

        # Prune triangles and vertices
    params.prune_triangles_threshold = 0.1

        # PARAMETER SECOND STAGE
    params.lr_triangles_points_init = 0.0015

    params.start_opacity_floor = 1000

    params.start_pruning = 4000
    params.sigma_until = 30000
    params.final_opacity_iter = 14000

    params.sigma_start = 17000

    params.splitt_large_triangles = 1000
    params.start_upsampling = 20000
    params.upscaling_factor = 1

    params.size_probs_zero = 7.5e-05
    params.size_probs_zero_image_space = 0.0

    params.prune_size = 1400

    return params

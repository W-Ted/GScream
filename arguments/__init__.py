#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

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
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._pretrained_model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.lod = 0

        self.specified_ply_path = ""
        self.load_mask = True
        self.load_depth = True
        self.load_norm = True
        self.load_midas = False # True
        self.is_spin = True
        self.is_ibr = False
        self.ref_image_path = ""
        self.ref_depth_path = ""
        self.ref_normal_path = ""
        self.ref_mask_path = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.uncertainty_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_uncertainty_lr_init = 0.002
        self.mlp_uncertainty_lr_final = 0.00002  
        self.mlp_uncertainty_lr_delay_mult = 0.01
        self.mlp_uncertainty_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000

        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000


        self.discriminator_lr_init = 0.01
        self.discriminator_lr_final = 0.00001
        self.discriminator_lr_delay_mult = 0.01
        self.discriminator_lr_max_steps = 30_000


        self.crossattn_lr_init = 0.01
        self.crossattn_lr_final = 0.00001
        self.crossattn_lr_delay_mult = 0.01
        self.crossattn_lr_max_steps = 30_000

        self.selfattn_lr_init = 0.01
        self.selfattn_lr_final = 0.00001
        self.selfattn_lr_delay_mult = 0.01
        self.selfattn_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_depth = 1.0
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        self.start_crossattn_from = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002


        # self.feature_lr = 0.0025
        # self.opacity_lr = 0.05
        # self.scaling_lr = 0.005
        # self.rotation_lr = 0.001
        # self.percent_dense = 0.01 # 调小

        self.lpips_lr = 0.01
        self.lpips_b  = 60
        # self.depth_lr = 0.01
        self.perceptual_lr = 0.01
        self.perceptual_b  = 60
        self.refer_rgb_lr = 1.0
        self.refer_rgb_lr_fg = 1.0
        self.other_rgb_lr = 1.0
        self.other_rgb_lr_fg = 0.0
        self.refer_depth_lr = 1.0
        self.refer_depth_lr_fg = 1.0
        self.refer_depth_lr_smooth = 1.0
        self.refer_depth_lr_smooth_edge = 1.0
        self.disp_smooth_lr = 1.0
        self.other_depth_lr = 1.0
        self.other_depth_lr_smooth = 1.0
        self.refer_normal_lr = 1.0
        self.other_normal_lr = 1.0
        self.refer_opacity_lr = 1.0
        self.other_opacity_lr = 1.0
        self.flat_lr = 1.0
        self.sparse_depth_lr = 0.0
        self.refer_warping_lr = 0.0

        self.vgg_lr = 0.0
        self.discriminator_lr = 0.0
        self.crossattn_lr = 0.0
        self.adv_lr = 0.0

        self.use_lama = False
        self.pretrained_ply = ''

        # cross-attn related
        self.enable_crossattn_refview = 1.0
        self.enable_crossattn_otherview = 1.0
        self.attn_head_num = 8
        self.attn_head_dim = 64

        # feat update related
        self.crossattn_feat_update_ema = 1.0
        self.enable_pe = 0.0

        # cross-attn sampling related
        self.enable_enlarge_samping = 0.0
        self.sampling_2D_enlarge_ratio = 1.1
        self.enable_edge_samping = 1.0
        self.enable_twopatch_samping = 0.0
        self.sampling_2D_small_ratio = 0.6

        # self-attn related
        self.enable_selfattn = 0.0
        self.selfattn_feat_update_ema = 1.0
        # self.selfattn_head_num = 8
        # self.selfattn_head_dim = 64
        self.selfattn_knn_maxnum = 2000
        self.selfattn_sampling_rad = 0.5 # need to be tuned

        self.crossattn_start_iter = 15000
        



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

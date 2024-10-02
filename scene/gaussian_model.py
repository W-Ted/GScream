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

import os
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch_scatter import scatter_max

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p

from scene.stylegan2 import Discriminator
import torch.nn.functional as F
from bidirectional_cross_attention import BidirectionalCrossAttention




class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.uncertainty_activation = torch.sigmoid
        self.inverse_uncertainty_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank = False,
                 attn_head_num = 8,
                 attn_head_dim = 64,
                 use_bidirectional_attn = False,
                 use_self_attn = False,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)
        self.uncertainty_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._uncertainty = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.optimizer_d = None
        self.optimizer_c = None
        self.optimizer_s = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.attn_head_num = attn_head_num
        self.attn_head_dim = attn_head_dim

        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()


        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_uncertainty = nn.Sequential(
            nn.Linear(feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            # nn.Tanh()
            nn.Sigmoid()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        # not used
        self.discriminator = Discriminator(
                block_kwargs={},
                mapping_kwargs={},
                epilogue_kwargs={"mbstd_group_size": 4},
                channel_base=16384,
                channel_max=512,
                num_fp16_res=4,
                conv_clamp=256,
                img_channels=3,
                c_dim=0,
                # img_resolution=64,
                img_resolution=256,
            ).cuda()
        
        if use_bidirectional_attn: # by default
            self.crossattn = BidirectionalCrossAttention(
                dim = 32,
                heads = self.attn_head_num,
                dim_head = self.attn_head_dim,
                context_dim = 32
            ).cuda()
        else: 
            self.crossattn = nn.MultiheadAttention(
                embed_dim=32, 
                num_heads=self.attn_head_num, 
                batch_first=True).cuda()
        self.EMB = Embedding(3, 6).cuda() # 36


        self.use_self_attn = use_self_attn

        if self.use_self_attn:
            self.selfattn = nn.MultiheadAttention(
                embed_dim=32, 
                num_heads=8, 
                batch_first=True).cuda()


    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_uncertainty.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_uncertainty.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._scaling,
            self._rotation,
            self._opacity,
            self._uncertainty,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.optimizer_d.state_dict(),
            self.optimizer_c.state_dict(),
            self.optimizer_s.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._uncertainty,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.optimizer_d.load_state_dict(opt_dict)
        self.optimizer_c.load_state_dict(opt_dict)
        self.optimizer_s.load_state_dict(opt_dict)


    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_uncertainty_mlp(self):
        return self.mlp_uncertainty
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def get_anchor_feat(self):
        return self._anchor_feat
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_uncertainty(self):
        return self.uncertainty_activation(self._uncertainty)
    
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # cal max xyz
        self.xyz_max = torch.max(fused_point_cloud.clone(), 0)[0].unsqueeze(0)
        print("MAX of points at initialisation : ", self.xyz_max)

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        uncertainties = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self._uncertainty = nn.Parameter(uncertainties.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.uncertainty_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._uncertainty], 'lr': training_args.uncertainty_lr, "name": "uncertainty"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_uncertainty.parameters(), 'lr': training_args.mlp_uncertainty_lr_init, "name": "mlp_uncertainty"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._uncertainty], 'lr': training_args.uncertainty_lr, "name": "uncertainty"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_uncertainty.parameters(), 'lr': training_args.mlp_uncertainty_lr_init, "name": "mlp_uncertainty"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

            l_d = [
                {'params': self.discriminator.parameters(), 'lr': training_args.discriminator_lr_init, "name": "discriminator"},
            ]

            l_c = [
                {'params': self.crossattn.parameters(), 'lr': training_args.crossattn_lr_init, "name": "crossattn"},
            ]

            if self.use_self_attn:
                l_s = [
                {'params': self.selfattn.parameters(), 'lr': training_args.selfattn_lr_init, "name": "selfattn"},
            ]



        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.optimizer_d = torch.optim.Adam(l_d, lr=0.0, eps=1e-15)
        self.optimizer_c = torch.optim.Adam(l_c, lr=0.0, eps=1e-15)
        if self.use_self_attn:
            self.optimizer_s = torch.optim.Adam(l_s, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_uncertainty_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_uncertainty_lr_init,
                                                    lr_final=training_args.mlp_uncertainty_lr_final,
                                                    lr_delay_mult=training_args.mlp_uncertainty_lr_delay_mult,
                                                    max_steps=training_args.mlp_uncertainty_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        self.discriminator_scheduler_args = get_expon_lr_func(lr_init=training_args.discriminator_lr_init,
                                                        lr_final=training_args.discriminator_lr_final,
                                                        lr_delay_mult=training_args.discriminator_lr_delay_mult,
                                                        max_steps=training_args.discriminator_lr_max_steps)
        
        self.crossattn_scheduler_args = get_expon_lr_func(lr_init=training_args.crossattn_lr_init,
                                                        lr_final=training_args.crossattn_lr_final,
                                                        lr_delay_mult=training_args.crossattn_lr_delay_mult,
                                                        max_steps=training_args.crossattn_lr_max_steps)
        if self.use_self_attn:
            self.selfattn_scheduler_args = get_expon_lr_func(lr_init=training_args.selfattn_lr_init,
                                                        lr_final=training_args.selfattn_lr_final,
                                                        lr_delay_mult=training_args.selfattn_lr_delay_mult,
                                                        max_steps=training_args.selfattn_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_uncertainty":
                lr = self.mlp_uncertainty_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr

        for param_group_d in self.optimizer_d.param_groups:
            if param_group_d["name"] == "discriminator":
                lr = self.discriminator_scheduler_args(iteration)
                param_group_d['lr'] = lr
        
        for param_group_c in self.optimizer_c.param_groups:
            if param_group_c["name"] == "crossattn":
                lr = self.crossattn_scheduler_args(iteration)
                param_group_c['lr'] = lr
        
        if self.use_self_attn:
            for param_group_s in self.optimizer_s.param_groups:
                if param_group_s["name"] == "selfattn":
                    lr = self.selfattn_scheduler_args(iteration)
                    param_group_s['lr'] = lr
        
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        l.append('uncertainty')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    


    def run_crossattn_pe_all_fix0228_00(self, fg_anchor_mask, bg_anchor_mask, pe=False, ema=1.0, is_ref=True):
        assert fg_anchor_mask.shape == bg_anchor_mask.shape

        self._anchor_feat = self._anchor_feat.detach()

        if not pe: # here
            fg_feat = self._anchor_feat[fg_anchor_mask][None,:,:].clone() # 1, N, 32
            bg_feat = self._anchor_feat[bg_anchor_mask][None,:,:].clone() # 1, N, 32
        # else:
            # fg_feat = self._anchor_feat[fg_anchor_mask][None,:,:] + self.EMB(2 * (self._anchor[fg_anchor_mask] / self.xyz_max) - 1).detach()[None,:,:32].clone()
            # bg_feat = self._anchor_feat[bg_anchor_mask][None,:,:] + self.EMB(2 * (self._anchor[bg_anchor_mask] / self.xyz_max) - 1).detach()[None,:,:32].clone()
            

        fg_feat_mask = torch.ones_like(fg_feat[:,:,0]).bool() # B, 1 -> 1, N
        bg_feat_mask = torch.ones_like(bg_feat[:,:,0]).bool() # B, 1 -> 1, N

        fg_feat_out, bg_feat_out = self.crossattn( 
            fg_feat,
            bg_feat,
            mask = fg_feat_mask,
            context_mask = bg_feat_mask
            )

        if is_ref: # only update the fg feature under reference view
            self._anchor_feat[fg_anchor_mask] = ema * fg_feat_out[0,:,:] + (1-ema) * self._anchor_feat[fg_anchor_mask] # P1 x 32
        # always update the bg feature
        self._anchor_feat[bg_anchor_mask] = ema * bg_feat_out[0,:,:] + (1-ema) * self._anchor_feat[bg_anchor_mask] # P2 x 32

        self._anchor_feat.retain_grad()

        return 


    

    def run_crossattn(self, fg_anchor_mask, bg_anchor_mask, pe=False, ema=1.0, is_ref=True):
        assert fg_anchor_mask.shape == bg_anchor_mask.shape

        self._anchor_feat = self._anchor_feat.detach()

        if not pe: # here
            fg_feat = self._anchor_feat[fg_anchor_mask][None,:,:].clone() # 1, N, 32
            bg_feat = self._anchor_feat[bg_anchor_mask][None,:,:].clone() # 1, N, 32
        # else:
            # fg_feat = self._anchor_feat[fg_anchor_mask][None,:,:] + self.EMB(2 * (self._anchor[fg_anchor_mask] / self.xyz_max) - 1).detach()[None,:,:32].clone()
            # bg_feat = self._anchor_feat[bg_anchor_mask][None,:,:] + self.EMB(2 * (self._anchor[bg_anchor_mask] / self.xyz_max) - 1).detach()[None,:,:32].clone()
            

        fg_feat_mask = torch.ones_like(fg_feat[:,:,0]).bool() # B, 1 -> 1, N
        bg_feat_mask = torch.ones_like(bg_feat[:,:,0]).bool() # B, 1 -> 1, N

        fg_feat_out, bg_feat_out = self.crossattn( 
            fg_feat,
            bg_feat,
            mask = fg_feat_mask,
            context_mask = bg_feat_mask
            )

        if is_ref: # only update the fg feature under reference view
            self._anchor_feat[fg_anchor_mask] = ema * fg_feat_out[0,:,:] + (1-ema) * self._anchor_feat[fg_anchor_mask] # P1 x 32
        # always update the bg feature
        self._anchor_feat[bg_anchor_mask] = ema * bg_feat_out[0,:,:] + (1-ema) * self._anchor_feat[bg_anchor_mask] # P2 x 32

        self._anchor_feat.retain_grad()

        return 



    def run_discriminator(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        bs, c, h, w = x.shape # [-1, 1]
        if h < 256 or w < 256:
            x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
        output = self.discriminator(x) # torch.Size([1, 1])
        return output


    def compute_adv_loss(self, fake_patch):
        pred_fake = self.run_discriminator(fake_patch)
        adv_loss = torch.nn.functional.softplus(-pred_fake).mean()
        return adv_loss

    def compute_simple_gradient_penalty(self, x):
        x.requires_grad_(True)
        pred_real = self.run_discriminator(x)
        gradients = torch.autograd.grad(outputs=[pred_real.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
        r1_penalty = gradients.square().sum([1, 2, 3]).mean()
        return r1_penalty / 2

    def compute_d_loss(self, real_patch, fake_patch):
        pred_real = self.run_discriminator(real_patch)
        pred_fake = self.run_discriminator(fake_patch.detach())
        loss_d_real = torch.nn.functional.softplus(-pred_real).mean()
        loss_d_fake = torch.nn.functional.softplus(pred_fake).mean()
        # gradient_penalty = self.config.gp_loss_mult * self.compute_simple_gradient_penalty(real_patch)
        # gradient_penalty = self.compute_simple_gradient_penalty(real_patch)
        # gradient_penalty = self.compute_simple_gradient_penalty(real_patch.unsqueeze(0))
        gradient_penalty = 0 #self.compute_simple_gradient_penalty(real_patch.unsqueeze(0))
        loss_d = loss_d_real + loss_d_fake
        return loss_d, gradient_penalty



    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        uncertainties = self._uncertainty.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, uncertainties, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)
        uncertainties = np.asarray(plydata.elements[0]["uncertainty"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._uncertainty = nn.Parameter(torch.tensor(uncertainties, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
    # def training_statis(self, viewspace_point_tensor, opacity, uncertainty, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)

        # #  update uncertainty stats
        # temp_uncertainty = uncertainty.clone().view(-1).detach()
        # temp_uncertainty[temp_uncertainty<0] = 0
        # temp_uncertainty = temp_uncertainty.view([-1, self.n_offsets])
        # self.uncertainty_accum[anchor_visible_mask] += temp_uncertainty.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._uncertainty = optimizable_tensors["uncertainty"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
                new_uncertainties = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                    "uncertainty": new_uncertainties,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                temp_uncertainty_accum = torch.cat([self.uncertainty_accum, torch.zeros([new_uncertainties.shape[0], 1], device='cuda').float()], dim=0)
                del self.uncertainty_accum
                self.uncertainty_accum = temp_uncertainty_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                self._uncertainty = optimizable_tensors["uncertainty"]
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.uncertainty_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_uncertainty_accum = self.uncertainty_accum[~prune_mask]
        del self.uncertainty_accum
        self.uncertainty_accum = temp_uncertainty_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self,path):
        mkdir_p(os.path.dirname(path))
        
        if self.use_feat_bank:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'uncertainty_mlp': self.mlp_uncertainty.state_dict(),
                'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict()}, path)
        else:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'uncertainty_mlp': self.mlp_uncertainty.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict()}, path)


    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_uncertainty.load_state_dict(checkpoint['uncertainty_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])




class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        # out = [x]
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)
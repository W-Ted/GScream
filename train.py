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
import numpy as np

import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim, ScaleAndShiftLoss, l1_loss_masked, ssim_masked, l2_loss, my_ssim, binary_cross_entropy_loss
from gaussian_renderer import prefilter_voxel, render, network_gui, prefilter_position2D
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import matplotlib.pyplot as plt
# from utils.loss_utils import compute_scale_and_shift

from utils.colmap_utils import read_cameras_binary
import cv2
import torch.nn.functional as F
import random
import time



# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
loss_fn_vgg_spatial = lpips.LPIPS(net='vgg', spatial=True).to('cuda')



def cal_box(mask):
    x_coords, y_coords = np.nonzero(mask[:, :]).split([1, 1], dim=1)
    x_coords = torch.squeeze(x_coords)
    y_coords = torch.squeeze(y_coords)
    x_min, y_min = torch.min(x_coords), torch.min(y_coords)
    x_max, y_max = torch.max(x_coords), torch.max(y_coords)
    return np.asarray([x_min.cpu(), y_min.cpu(), x_max.cpu(), y_max.cpu()])


def expand_bbox_by_ratio(bbox_ori, enlarge_ratio, max_h, max_w, int_=False):
    x_min, y_min, x_max, y_max = bbox_ori[0], bbox_ori[1], bbox_ori[2], bbox_ori[3]

    # calculate ori bbox's w h
    width = x_max - x_min
    height = y_max - y_min

    # calculate ori bbox's center coor
    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2

    # calculate new bbox's w h
    new_width = width * enlarge_ratio
    new_height = height * enlarge_ratio

    # calculate new bbox's x_min, y_min, x_max, y_max
    new_x_min = center_x - new_width / 2
    new_y_min = center_y - new_height / 2
    new_x_max = center_x + new_width / 2
    new_y_max = center_y + new_height / 2

    if not int_:
        new_x_min = max(0, new_x_min)
        new_y_min = max(0, new_y_min)
        new_x_max = min(new_x_max, max_h-1)
        new_y_max = min(new_y_max, max_w-1)

        # return coor of new bbox
        return new_x_min, new_y_min, new_x_max, new_y_max, width, height, new_width, new_height
    else:
        new_x_min = max(0, int(new_x_min))
        new_y_min = max(0, int(new_y_min))
        new_x_max = min(int(new_x_max), max_h-1)
        new_y_max = min(int(new_y_max), max_w-1)

        # return coor of new bbox
        return new_x_min, new_y_min, new_x_max, new_y_max, width, height, new_width, new_height


def get_random_mask(gt_mask, enlarge_ratio, small_ratio, max_h, max_w):
    bbox_ori = cal_box(gt_mask) # 0 0 h w
    new_x_min, new_y_min, new_x_max, new_y_max, ori_width, ori_height, new_width, new_height = expand_bbox_by_ratio(bbox_ori, enlarge_ratio=enlarge_ratio, max_h=max_h, max_w=max_w) # new_x_min, new_y_min, new_x_max, new_y_max
    small_width, small_height = small_ratio * ori_width, small_ratio * ori_height

    # calculate bbox's max width & height
    max_small_width = new_width - small_width
    max_small_height = new_height - small_height

    # generate random offset
    offset_x = random.uniform(0, max_small_width)
    offset_y = random.uniform(0, max_small_height)

    # calculate bbox's new x_min, y_min, x_max, y_max
    new_x_min = max(0, int(new_x_min + offset_x))
    new_y_min = max(0, int(new_y_min + offset_y))
    new_x_max = min(int(new_x_min + small_width), max_h-1)
    new_y_max = min(int(new_y_min + small_height), max_w-1)

    fg_mask = torch.zeros_like(gt_mask).unsqueeze(0)
    fg_mask[:, new_x_min:new_x_max, new_y_min:new_y_max] = 1.0
    return fg_mask


# sampling for whole image 
def sample_patch_in_whole_image(H, W, patch_size=256):
    if H < patch_size or W < patch_size:
        raise ValueError("Input dimensions should be larger than the patch size.")

    x_min = torch.randint(0, W - patch_size + 1, (1,))
    y_min = torch.randint(0, H - patch_size + 1, (1,))
    x_max = x_min + patch_size
    y_max = y_min + patch_size

    # return x_min.item(), y_min.item(), x_max.item(), y_max.item()
    return y_min.item(), y_max.item(), x_min.item(), x_max.item()


def expand_bbox_256_bysize(bbox_ori, patch_size, max_h, max_w):
    x_min, y_min, x_max, y_max = bbox_ori[0], bbox_ori[1], bbox_ori[2], bbox_ori[3]

    # calculate ori bbox's w h
    width = x_max - x_min
    height = y_max - y_min

    # calculate ori bbox's center coor
    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2

    # calculate new bbox's w h
    new_width = patch_size
    new_height = patch_size

    # calculate new bbox's x_min, y_min, x_max, y_max
    new_x_min = center_x - new_width / 2
    new_y_min = center_y - new_height / 2
    new_x_max = center_x + new_width / 2
    new_y_max = center_y + new_height / 2

    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(new_x_max, max_h-1)
    new_y_max = min(new_y_max, max_w-1)

    # return coor of new bbox
    return new_x_min, new_y_min, new_x_max, new_y_max, width, height, new_width, new_height


def sample_patch_in_mask_region(gt_mask, patch_size, small_ratio, max_h, max_w):
    bbox_ori = cal_box(gt_mask) # 0 0 h w
    new_x_min, new_y_min, new_x_max, new_y_max, ori_width, ori_height, new_width, new_height = expand_bbox_256_bysize(bbox_ori, patch_size=256, max_h=max_h, max_w=max_w) # new_x_min, new_y_min, new_x_max, new_y_max
    small_width, small_height = small_ratio * ori_width, small_ratio * ori_height

    # calculate bbox's max width & height
    max_small_width = new_width - small_width 
    max_small_height = new_height - small_height 

    # generate random offset
    offset_x = random.uniform(0, max_small_width) 
    offset_y = random.uniform(0, max_small_height) 

    # calculate bbox's new x_min, y_min, x_max, y_max
    new_x_min = max(0, int(new_x_min + offset_x))
    new_y_min = max(0, int(new_y_min + offset_y))
    new_x_max = min(int(new_x_min + patch_size), max_h-1)
    new_y_max = min(int(new_y_min + patch_size), max_w-1)

    return new_x_min, new_x_max, new_y_min, new_y_max






# # copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def gradient_loss(prediction, target, mask, reduction=reduction_image_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


def depth2pcd_fromplane(depth2d, c2w, K, h, w):
    # pixel location
    i, j = torch.meshgrid(torch.linspace(0, w - 1, w), torch.linspace(0, h - 1, h),
                          indexing='ij')  # pytorch's meshgrid has indexing='ij'
    i = i.t().cuda() # yy
    j = j.t().cuda() # xx
    x_normalized = (i - K[0][2]) / K[0][0]
    y_normalized = (j - K[1][2]) / K[1][1]
    # camera space
    x_camera = x_normalized * depth2d
    y_camera = y_normalized * depth2d
    z_camera = depth2d
    xyz1_camera = torch.concat([x_camera, y_camera, z_camera, torch.ones_like(x_camera)], 0).reshape(4,h*w) # 
    # print(c2w.shape, xyz1_camera.shape)
    xyz_fine = torch.matmul(c2w, xyz1_camera).reshape(1,3,h,w) # (1 3 h w)
    return xyz_fine


def least_square_normal_regress_fast01(x_depth3d, global_eye, global_b, size=9, gamma=0.15, depth_scaling_factor=1, eps=1e-5):    
    
    stride=1
    xyz_padded = F.pad(x_depth3d, (size//2, size//2, size//2, size//2), mode='replicate')
    xyz_patches = xyz_padded.unfold(2, size, stride).unfold(3, size, stride) # [batch_size, 3, width, height, size, size]
    xyz_patches = xyz_patches.reshape((*xyz_patches.shape[:-2], ) + (-1,))  # [batch_size, 3, width, height, size*size]
    xyz_perm = xyz_patches.permute([0, 2, 3, 4, 1])
    
    diffs = xyz_perm - xyz_perm[:, :, :, (size*size)//2].unsqueeze(3)
    diffs = diffs / xyz_perm[:, :, :, (size*size)//2].unsqueeze(3)
    diffs[..., 0] = diffs[..., 2]
    diffs[..., 1] = diffs[..., 2]
    xyz_perm[torch.abs(diffs) > gamma] = 0.0


    batch_size, width, height, size, _ = xyz_perm.shape

    # Manual pseudoinverse
    xyz_perm = xyz_perm.view(batch_size * width * height, size, 3)
    A = torch.bmm(xyz_perm.transpose(1,2), xyz_perm)
    A_det = torch.det(A)
    A[A_det < eps, :, :] = global_eye

    A_inv = torch.inverse(A) #.view(-1,3,3)
    lstsq = A_inv.matmul(xyz_perm.transpose(1,2)).matmul(global_b).squeeze() # hw, 3
    lstsq = torch.nn.functional.normalize(lstsq, dim=1)
    lstsq[torch.isnan(lstsq)] = 0.0

    return -lstsq.view([1,height,width,3]).permute([0, 3, 1, 2])


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, opt.attn_head_num, opt.attn_head_dim, use_bidirectional_attn=True)
    scene = Scene(dataset, gaussians, ply_path=ply_path) # dataloader
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ### load ref
    all_image_names = [i for i in sorted(os.listdir(dataset.source_path+'/images_4')) if i.endswith('png')]
    ref_name = all_image_names[40+29][:-4] # for spinnerf dataset
    viewpoint_stack_first = scene.getTrainCameras().copy()
    viewpoint_ref = [vp for vp in viewpoint_stack_first if vp.image_name == ref_name]
    assert len(viewpoint_ref) == 1, len(viewpoint_ref)
    viewpoint_ref = viewpoint_ref[0]

    # load ref image
    ref_image_path = dataset.ref_image_path
    assert os.path.exists(ref_image_path), ref_image_path
    assert ref_image_path.endswith('png'), ref_image_path
    ref_image = torch.from_numpy(cv2.imread(ref_image_path)[:,:,[2,1,0]].astype(np.float32)) / 255.
    ref_image = ref_image.permute(2, 0, 1).cuda() # read unaligned depth 

    # load ref depth
    ref_depth_path = dataset.ref_depth_path
    assert os.path.exists(ref_depth_path), ref_depth_path
    assert ref_depth_path.endswith('npy'), ref_depth_path
    ref_depth = torch.from_numpy(np.load(ref_depth_path).astype(np.float32)).unsqueeze(0).cuda() # read unaligned depth 

    # load K
    camdata = read_cameras_binary(os.path.join(dataset.source_path, 'sparse/0/cameras.bin'))
    H = camdata[1].height
    W = camdata[1].width
    img_wh = [W/4, H/4]
    h, w = int(img_wh[1]), int(img_wh[0])
    focal = camdata[1].params[0] * img_wh[0]/W 
    K = np.array([
    [focal, 0, 0.5 * img_wh[0]],
    [0, focal, 0.5 * img_wh[1]],
    [0, 0, 1]
    ])
    K = torch.from_numpy(K.astype(np.float32)).cuda()
    print('K: ', K)


    #### load ref
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    torch.autograd.set_detect_anomaly(True)
    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # gt preparation
        gt_image = viewpoint_cam.original_image.cuda() # 3,h,w
        gt_mask = viewpoint_cam.original_mask.cuda()   # 1,h,w, fg = 1., bg = 0.
        gt_image_name = viewpoint_cam.image_name 
        if gt_image_name == ref_name:
            midas_depth = ref_depth
            gt_image = ref_image
        else:
            midas_depth = viewpoint_cam.midas_depth.cuda()

        # get mask: 
        voxel_visible_mask, position2D_x, position2D_y = prefilter_position2D(viewpoint_cam, gaussians, pipe, background)

        # valid2D_mask
        valid_mask_y = torch.logical_and(position2D_y>0, position2D_y<h)
        valid_mask_x = torch.logical_and(position2D_x>0, position2D_x<w)
        valid_mask_2D = torch.logical_and(valid_mask_x, valid_mask_y)
        valid_mask_2D = torch.logical_and(valid_mask_2D, voxel_visible_mask) # anchor points, which splatting to this view
        position2D_y = position2D_y.long()
        position2D_x = position2D_x.long()

        # if ((opt.enable_crossattn_refview > 0) and (gt_image_name == ref_name) and iteration > 15000) \
        #     or ((opt.enable_crossattn_otherview > 0) and (gt_image_name != ref_name) and iteration > 15000):
        if ((opt.enable_crossattn_refview > 0) and (gt_image_name == ref_name) and iteration > opt.start_crossattn_from) \
            or ((opt.enable_crossattn_otherview > 0) and (gt_image_name != ref_name) and iteration > opt.start_crossattn_from):

            try:
            # if True:
                if opt.enable_edge_samping > 0:
                    min_y, max_y, min_x, max_x = sample_patch_in_mask_region(gt_mask=gt_mask[0], patch_size=256, small_ratio=opt.sampling_2D_small_ratio, max_h=h, max_w=w)
                else:
                    print('No sampling method. ')
                
                sampled_mask_2D = torch.zeros_like(gt_mask[0]).long()
                sampled_mask_2D[min_y:max_y, min_x:max_x] = 1.0 

                anchor_2Dlabel_sample = sampled_mask_2D[position2D_y[valid_mask_2D], position2D_x[valid_mask_2D]] # (projected) anchor's 2D label
                num_sampled = anchor_2Dlabel_sample.sum()
                if not (num_sampled > 0):
                    print('num_sampled <= 0, %d'%num_sampled)
                    exit()

                voxel_sampled_mask = -1 * torch.ones_like(voxel_visible_mask) # -1
                voxel_sampled_mask[valid_mask_2D] = anchor_2Dlabel_sample # -1: invalid, 0: unampled, 1: sampled

                anchor_2Dlabel_fgbg = gt_mask[0].long()[position2D_y[valid_mask_2D], position2D_x[valid_mask_2D]] # (projected) anchor's 2D label
                voxel_fgbg_mask = -1 * torch.ones_like(voxel_visible_mask) # -1
                voxel_fgbg_mask[valid_mask_2D] = anchor_2Dlabel_fgbg # -1: invalid, 0: bg, 1: fg

                # 2-category cluster 
                fgbg_sampled = voxel_fgbg_mask[voxel_sampled_mask>0] # P 
                anchor_sampled = gaussians.get_anchor[voxel_sampled_mask>0] # P, 3

                anchor_sampled_fg_xyz = anchor_sampled[fgbg_sampled>0] # P1 x 3
                anchor_sampled_fg_indices = torch.nonzero(fgbg_sampled>0).squeeze() # P1

                anchor_sampled_bg_xyz = anchor_sampled[fgbg_sampled==0] # P2 x 3
                anchor_sampled_bg_indices = torch.nonzero(fgbg_sampled==0).squeeze() # P2

                if (anchor_sampled.shape[0] <= 11):
                    print('anchor_sampled <= 11, %d'%num_sampled)
                    exit()
                
                if (anchor_sampled_fg_xyz.shape[0] <= 11):
                    print('anchor_sampled <= 11, %d'%num_sampled)
                    exit()
                if (anchor_sampled_bg_xyz.shape[0] <= 11):
                    print('anchor_sampled <= 11, %d'%num_sampled)
                    exit()

                row_indices = anchor_sampled_fg_indices
                sampled_indices = anchor_sampled_bg_indices

                if not (sampled_indices.shape[0] > 0):
                    print('number pair <=0, %d'%sampled_indices.shape[0])
                    exit()

                min_num = min(min(sampled_indices.shape[0], row_indices.shape[0]), 2000)
                unique_sampled_indices = sampled_indices[torch.randperm(sampled_indices.size(0))][:min_num]
                unique_row_indices = row_indices[torch.randperm(row_indices.size(0))][:min_num]

                voxel_sampled_src = torch.zeros_like(voxel_visible_mask) # P
                voxel_used_src = torch.zeros(num_sampled).cuda().bool()
                voxel_used_src[unique_row_indices] = True
                voxel_sampled_src[voxel_sampled_mask>0] = voxel_used_src

                voxel_sampled_dst = torch.zeros_like(voxel_visible_mask)
                voxel_used_dst = torch.zeros(num_sampled).cuda().bool()
                voxel_used_dst[unique_sampled_indices] = True
                voxel_sampled_dst[voxel_sampled_mask>0] = voxel_used_dst # P 

                # 3. choose
                gaussians.run_crossattn(voxel_sampled_src, voxel_sampled_dst, pe=(opt.enable_pe>0), ema=opt.crossattn_feat_update_ema, is_ref=(gt_image_name == ref_name))
                cross_flag = True

            except:
            # else:
                print('No valid sampled anchors...')
                cross_flag = False
                continue
        else:
            cross_flag = False

        retain_grad = (iteration < opt.update_until and iteration >= 0)
        # render with mask
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity =\
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        # render_uncertainty = render_pkg["uncertainty"]
        
        # RGB loss
        if gt_image_name == ref_name:
            print('Using reference view... ')
            Ll1 = l1_loss(image, gt_image)
            loss = opt.refer_rgb_lr * ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
            if opt.refer_rgb_lr_fg > opt.refer_rgb_lr:
                loss += (opt.refer_rgb_lr_fg - opt.refer_rgb_lr) * ((1.0 - opt.lambda_dssim) * l1_loss_masked(image, gt_image, gt_mask) + opt.lambda_dssim * (1.0 - ssim_masked(image, gt_image, gt_mask)))
        else:
            other_rgb_weight = (1.0-gt_mask) + opt.other_rgb_lr_fg * gt_mask
            Ll1 = l1_loss_masked(image, gt_image, other_rgb_weight)
            loss = opt.other_rgb_lr * ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_masked(image, gt_image, other_rgb_weight)))

        # Depth loss
        if opt.refer_depth_lr > 0 and gt_image_name == ref_name:
            depth = render_pkg["render_depth"]
            valid_mask = 1.0-gt_mask 
            scale, shift = compute_scale_and_shift(depth, midas_depth, valid_mask)
            scale = torch.abs(scale) # for excluding negative scale 
            aligned_depth = scale.view(-1, 1, 1) * depth + shift.view(-1, 1, 1)
            loss += opt.refer_depth_lr * l1_loss(aligned_depth, midas_depth)
            if opt.refer_depth_lr_fg > opt.refer_depth_lr:
                fg_mask = get_random_mask(gt_mask[0], enlarge_ratio=1.5, small_ratio=0.8, max_h=h, max_w=w)
                loss += (opt.refer_depth_lr_fg - opt.refer_depth_lr) * l1_loss_masked(aligned_depth, midas_depth, fg_mask)
            for scale in range(4):
                step = pow(2, scale)
                loss += 0.5 * opt.refer_depth_lr_smooth * gradient_loss(aligned_depth[:,::step,::step], midas_depth[:,::step,::step], torch.ones_like(gt_mask)[:,::step,::step])

        # 2.2 depth loss from other view
        if opt.other_depth_lr > 0 and gt_image_name != ref_name:
            depth = render_pkg["render_depth"]
            valid_mask = (1.0-gt_mask) 
            scale, shift = compute_scale_and_shift(depth, midas_depth, valid_mask)
            scale = torch.abs(scale) # modi here
            aligned_depth = scale.view(-1, 1, 1) * depth + shift.view(-1, 1, 1)
            loss += opt.other_depth_lr * l1_loss_masked(aligned_depth, midas_depth, valid_mask)

            for scale in range(4):
                step = pow(2, scale)
                # loss += 0.5 * opt.other_depth_lr * gradient_loss(aligned_depth[:,::step,::step], midas_depth[:,::step,::step], (valid_mask)[:,::step,::step])
                loss += 0.5 * opt.other_depth_lr_smooth * gradient_loss(aligned_depth[:,::step,::step], midas_depth[:,::step,::step], (valid_mask)[:,::step,::step])
        
        loss.backward(retain_graph=True) # 200 


        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations): 
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration) 
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat: # 500 - 15000
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0: # > 1500， 100
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                if cross_flag:
                    gaussians.optimizer_c.step()
                    gaussians.optimizer_c.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask, _, _ = prefilter_position2D(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, source_path=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    if name !='spiral':
        error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    else:
        rgbd_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rgbd")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_renders")
    uncertainty_path = os.path.join(model_path, name, "ours_{}".format(iteration), "uncertainty_renders")

    makedirs(render_path, exist_ok=True)
    if name !='spiral':
        makedirs(error_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
    else:
        makedirs(rgbd_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(uncertainty_path, exist_ok=True)

    if name == 'spiral':
        RGBD = []
        camdata = read_cameras_binary(os.path.join(source_path, 'sparse/0/cameras.bin'))
        H = camdata[1].height
        W = camdata[1].width
        img_wh = [W/4, H/4]
        h, w = int(img_wh[1]), int(img_wh[0])
        focal = camdata[1].params[0] * img_wh[0]/W 
        K = np.array([
        [focal, 0, 0.5 * img_wh[0]],
        [0, focal, 0.5 * img_wh[1]],
        [0, 0, 1]
        ])
        K = torch.from_numpy(K.astype(np.float32)).cuda()
        print('K: ', K)

        # for norm calculation
        kernel_size = 9
        global_eye = torch.eye(3).cuda()
        global_b = torch.ones([h*w,kernel_size**2,1]).cuda()
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        # voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        voxel_visible_mask, _, _ = prefilter_position2D(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        # uncertainty
        uncertainty = torch.clamp(render_pkg["uncertainty"], 0.0, 1.0)
        
        # add depth rendering

        if name != 'spiral':
            render_depth = render_pkg["render_depth"]
            depth_ori = render_depth.clone()

            # gt_depth = view.depth
            gt_depth = view.midas_depth
            gt_mask = view.original_mask
            valid_mask = 1 - gt_mask
            scale, shift = compute_scale_and_shift(render_depth, gt_depth, valid_mask)
            scale = torch.abs(scale)
            depth = render_depth * scale + shift

            depth_concat = torch.cat((depth, gt_depth), dim=0).unsqueeze(1)
            tensor = torchvision.utils.make_grid(depth_concat, padding=0, normalize=False, scale_each=False).cpu().detach().numpy()
            plt.imsave(os.path.join(depth_path, '{0:05d}'.format(idx) + "_depth.png"), np.transpose(tensor, (1,2,0))[:,:,0], cmap="viridis")
            # gts
            gt = view.original_image[0:3, :, :]
            
            # error maps
            errormap = (rendering - gt).abs()

            name_list.append('{0:05d}'.format(idx) + ".png")
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(uncertainty, os.path.join(uncertainty_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
        
        else:
            depth = render_pkg["render_depth"]
            depth_ori = depth.clone()

            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = (255 * depth.cpu().numpy()).astype(np.uint8)[0]
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:,:,[2,1,0]]
            depth = torch.FloatTensor(depth).permute([2,0,1]) / 255.

            rendered = torch.flip(rendering.unsqueeze(0), dims=[3])[0]
            uncertainty = torch.flip(uncertainty.unsqueeze(0), dims=[3])[0] # 1 h w
            uncertainty = uncertainty.expand([3, -1, -1])
            depth = torch.flip(depth.unsqueeze(0), dims=[3])[0]

            # add normal
            c2w = view.C2W[:3,:4]
            c2w = torch.from_numpy(c2w.astype(np.float32)).cuda()
            depth_map_3d = depth2pcd_fromplane(depth2d=depth_ori, c2w=c2w, K=K, h=h, w=w)
            normal = least_square_normal_regress_fast01(depth_map_3d, global_eye, global_b)
            normal = normal.reshape(3,h,w).permute([1,2,0])
            normal = (torch.nn.functional.normalize(normal, p=2, dim=-1)).permute([2,0,1]) # 3, h, w
            normal = (normal + 1) / 2
            normal = torch.flip(normal.unsqueeze(0), dims=[3])[0]

            rgbd = torch.concat([rendered.cpu(), depth, normal.cpu(), uncertainty.cpu()], 2)
            torchvision.utils.save_image(rendered, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(uncertainty, os.path.join(uncertainty_path, '{0:05d}'.format(idx) + ".png"))
            if name != 'spiral':
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            else:
                torchvision.utils.save_image(rgbd, os.path.join(rgbd_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        
        if name == 'train':
            np.save(os.path.join(depth_path, '{0:05d}'.format(idx) + ".npy"), depth_ori.cpu().numpy())


    if name != 'spiral':
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
                    json.dump(per_view_dict, fp, indent=True)
    else:
        cmd = f"/usr/local/bin/ffmpeg -y -i {rgbd_path}/%05d.png -q:v 0 {os.path.join(model_path, name,'-'.join(model_path.split('/')[1:-1])+'.mp4')}"
        print(cmd)
        os.system(cmd)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank)
        dataset.pretrained_model_path = ""
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        t_spiral_list, visible_count = render_set(dataset.model_path, "spiral", scene.loaded_iter, scene.getSpiralCameras(), gaussians, pipeline, background, dataset.source_path)
        spiral_fps = 1.0 / torch.tensor(t_spiral_list[5:]).mean()
        logger.info(f'Spiral FPS: \033[1;35m{spiral_fps.item():.5f}\033[0m')
        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/spiral_FPS', spiral_fps.item(), 0)
        if wandb is not None:
            wandb.log({"spiral_fps":spiral_fps, })
        
        if not skip_train:
            t_train_list, _  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir, mask_dir):
    renders = []
    gts = []
    image_names = []
    masks = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)

        mask = Image.open(os.path.join(mask_dir, 'out_%05d.png'%(1+int(fname.split('.')[0]))))
        mask = mask.resize((1008, 567), Image.LANCZOS)
        masks.append(tf.to_tensor(mask).bool().expand([3,-1,-1]).unsqueeze(0).cuda())
    return renders, gts, image_names, masks


def evaluate(dataset, model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"
    mask_dir = Path(dataset.source_path) / "images_4" / "test_label"


    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names, masks = readImages(renders_dir, gt_dir, mask_dir)

        ssims = []
        psnrs = []
        lpipss = []

        masked_ssims = []
        masked_psnrs = []
        masked_lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            # print(renders[idx].shape, renders[idx].dtype, renders[idx].min(), renders[idx].max())
            # print(gts[idx].shape, gts[idx].dtype, gts[idx].min(), gts[idx].max())
            ssims.append(my_ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(2*renders[idx]-1, 2*gts[idx]-1).detach())

            masked_ssims.append(my_ssim(renders[idx], gts[idx], masks[idx]))
            masked_psnrs.append(psnr(renders[idx], gts[idx], masks[idx]))
            cur_lpips_spatial = loss_fn_vgg_spatial(2*renders[idx]-1, 2*gts[idx]-1).squeeze() 
            masked_lpipss.append(cur_lpips_spatial[masks[idx][0,0]].mean())

        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })
            wandb.log({"test_masked_SSIMS":torch.stack(masked_ssims).mean().item(), })
            wandb.log({"test_masked_PSNR_final":torch.stack(masked_psnrs).mean().item(), })
            wandb.log({"test_masked_LPIPS":torch.stack(masked_lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        logger.info("  masked SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(masked_ssims).mean(), ".5"))
        logger.info("  masked PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(masked_psnrs).mean(), ".5"))
        logger.info("  masked LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(masked_lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)
    logger = get_logger(model_path)

    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')
    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.") 

    # calculate metrics
    logger.info("\n Starting evaluation...")
    evaluate(lp.extract(args), args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.") 

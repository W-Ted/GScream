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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    # def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, mask, depth, normal,
    # def __init__(self, colmap_id, R, T, FoVx, FoVy, cx, cy, image, gt_alpha_mask, mask, depth, normal,
    # def __init__(self, colmap_id, R, T, FoVx, FoVy, cx, cy, image, gt_alpha_mask, mask, depth, sparse_depth, normal,
    def __init__(self, colmap_id, R, T, FoVx, FoVy, cx, cy, image, gt_alpha_mask, mask, depth, normal,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx = cx
        self.cy = cy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.original_mask = mask.to(self.data_device)
        self.midas_depth = depth.to(self.data_device)
        # self.sparse_depth = sparse_depth.to(self.data_device)
        self.omni_normal = normal.to(self.data_device)

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans # np.array([0.0, 0.0, 0.0])
        self.scale = scale # 1.0

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda() # w2c.T
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=self.cx, cy=self.cy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.view_world_transform = self.world_view_transform.transpose(0, 1).inverse() # c2w
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class Camera_spiral(nn.Module):
    # def __init__(self, colmap_id, R, T, C2W, FoVx, FoVy, image_height, image_width, image, gt_alpha_mask, mask,
    def __init__(self, colmap_id, R, T, C2W, FoVx, FoVy, cx, cy, image_height, image_width, image, gt_alpha_mask, mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera_spiral, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx = cx
        self.cy = cy
        self.image_name = image_name

        self.C2W = C2W

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        # self.image_width = self.original_image.shape[2]
        # self.image_height = self.original_image.shape[1]

        self.image_width = image_width
        self.image_height = image_height

        # self.original_mask = mask.to(self.data_device)

        # if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
        # else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale


        rescale = False
        if rescale:
            cam_center = C2W[:3, 3]
            cam_center = (cam_center + trans) * scale
            C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        Rt = np.float32(Rt)

        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.world_view_transform = torch.tensor(Rt).transpose(0, 1).cuda()
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=self.cx, cy=self.cy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # print('Loading Spiral')



class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]




# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #

# import torch
# from torch import nn
# import numpy as np
# from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# class Camera(nn.Module):
#     def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
#                  image_name, depth, normal, uid,
#                  trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
#                  ):
#         super(Camera, self).__init__()

#         self.uid = uid
#         self.colmap_id = colmap_id
#         self.R = R
#         self.T = T
#         self.FoVx = FoVx
#         self.FoVy = FoVy
#         self.image_name = image_name
        
#         try:
#             self.data_device = torch.device(data_device)
#         except Exception as e:
#             print(e)
#             print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
#             self.data_device = torch.device("cuda")

#         self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
#         self.image_width = self.original_image.shape[2]
#         self.image_height = self.original_image.shape[1]
#         # depth and normal
#         self.depth = depth.to(self.data_device)
#         self.normal = normal.to(self.data_device)

#         if gt_alpha_mask is not None:
#             self.original_image *= gt_alpha_mask.to(self.data_device)
#             self.depth *= gt_alpha_mask.to(self.data_device)
#             self.normal *= gt_alpha_mask.to(self.data_device)
#         else:
#             self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
#             self.depth *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
#             self.normal *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

#         self.zfar = 100.0
#         self.znear = 0.01

#         self.trans = trans
#         self.scale = scale

#         self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
#         self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
#         self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
#         self.camera_center = self.world_view_transform.inverse()[3, :3]

# class MiniCam:
#     def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
#         self.image_width = width
#         self.image_height = height    
#         self.FoVy = fovy
#         self.FoVx = fovx
#         self.znear = znear
#         self.zfar = zfar
#         self.world_view_transform = world_view_transform
#         self.full_proj_transform = full_proj_transform
#         view_inv = torch.inverse(self.world_view_transform)
#         self.camera_center = view_inv[3][:3]


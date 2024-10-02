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

from scene.cameras import Camera, Camera_spiral
import numpy as np
from utils.general_utils import PILtoTorch, NPtoTorch, PILtoTorch_01mask, PILtoTorch_depth, PILtoTorch_normal
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    # resize_depth = NPtoTorch(cam_info.depth)
    # resize_normal = NPtoTorch(cam_info.normal)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    # print(f'gt_image: {gt_image.shape}')
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    # return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
    #               FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
    #               image=gt_image, gt_alpha_mask=loaded_mask,
    #               image_name=cam_info.image_name, 
    #               depth=resize_depth, normal=resize_normal, uid=id, data_device=args.data_device)
        
    if not args.load_mask:
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=gt_image, gt_alpha_mask=loaded_mask,
                    image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    elif not args.load_depth: # load mask, not load depth
        resized_mask = PILtoTorch_01mask(cam_info.mask, resolution)
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=gt_image, gt_alpha_mask=loaded_mask, mask=resized_mask, 
                    image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    else: # load mask and depth
        resized_mask = PILtoTorch_01mask(cam_info.mask, resolution)
        resized_depth = PILtoTorch_depth(cam_info.depth, resolution)
        # print(cam_info.sparse_depth)
        # resized_sparse_depth = PILtoTorch_depth(cam_info.sparse_depth, resolution)
        if not args.load_norm:
            # return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
            #             FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
            #             image=gt_image, gt_alpha_mask=loaded_mask, mask=resized_mask, depth=resized_depth, 
            #             image_name=cam_info.image_name, uid=id, data_device=args.data_device)


            # 1112 just debug here:
            return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                        FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                        image=gt_image, gt_alpha_mask=loaded_mask, mask=resized_mask, depth=resized_depth, normal=gt_image,
                        image_name=cam_info.image_name, uid=id, data_device=args.data_device)
        else:
            resized_normal = PILtoTorch_normal(cam_info.normal, resolution)
            # return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
            #             FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
            #             image=gt_image, gt_alpha_mask=loaded_mask, mask=resized_mask, depth=resized_depth, normal=resized_normal,
            #             image_name=cam_info.image_name, uid=id, data_device=args.data_device)
            # return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
            #             FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
            #             cx=cam_info.cx, cy=cam_info.cy,
            #             image=gt_image, gt_alpha_mask=loaded_mask, mask=resized_mask, depth=resized_depth, normal=resized_normal,
            #             image_name=cam_info.image_name, uid=id, data_device=args.data_device)
            return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                        FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                        cx=cam_info.cx, cy=cam_info.cy,
                        # image=gt_image, gt_alpha_mask=loaded_mask, mask=resized_mask, depth=resized_depth, sparse_depth=resized_sparse_depth, normal=resized_normal,
                        image=gt_image, gt_alpha_mask=loaded_mask, mask=resized_mask, depth=resized_depth, normal=resized_normal,
                        image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


# def loadCam_spiral(args, id, c2w, h, w, FovX, FovY, resolution_scale):
def loadCam_spiral(args, id, c2w, h, w, FovX, FovY, cx, cy, resolution_scale):
    # orig_w, orig_h = cam_info.image.size
    orig_w, orig_h = h, w

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

        image_height, image_width = resolution[1], resolution[0]

    # resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    # gt_image = resized_image_rgb[:3, ...]
    # loaded_mask = None

    # if resized_image_rgb.shape[1] == 4:
        # loaded_mask = resized_image_rgb[3:4, ...]



    if not args.load_mask:
        return Camera_spiral(colmap_id=None, R=None, T=None, c2w=c2w,
                    FoVx=FovX, FoVy=FovY, cx=cx, cy=cy, image_height=image_height, image_width=image_width,
                    image=None, gt_alpha_mask=None,
                    image_name=None, uid=id, data_device=args.data_device)
    else:
        # resized_mask = PILtoTorch(cam_info.mask, resolution)
        # return Camera_spiral(colmap_id=None, R=None, T=None, C2W=c2w,
        #             FoVx=FovX, FoVy=FovY, image_height=image_height, image_width=image_width,
        #             image=None, gt_alpha_mask=None, mask=None, 
        #             image_name=None, uid=id, data_device=args.data_device)
        return Camera_spiral(colmap_id=None, R=None, T=None, C2W=c2w,
                    FoVx=FovX, FoVy=FovY, cx=cx, cy=cy, image_height=image_height, image_width=image_width,
                    image=None, gt_alpha_mask=None, mask=None, 
                    image_name=None, uid=id, data_device=args.data_device)


def cameraList_from_camInfos_spiral(c2ws, h, w, FovX, FovY, cx, cy, resolution_scale, args):
    camera_list = []
    for id, c2w in enumerate(c2ws):
        camera_list.append(loadCam_spiral(args, id, c2w, h, w, FovX, FovY, cx, cy, resolution_scale))
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry



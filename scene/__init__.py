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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_camInfos_spiral
from utils.colmap_utils import read_images_binary, read_cameras_binary

import torch
import numpy as np

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None, rads_scale=0.15):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.pretrained_model_path = args.pretrained_model_path if args.pretrained_model_path != "" else args.model_path  #
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.pretrained_model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.spiral_cameras = {}

        print('source path: ', args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.specified_ply_path != "" and args.specified_ply_path.endswith('ply'): # here
                scene_info = sceneLoadTypeCallbacks["Colmap_ply"](args.source_path, args.images, args.eval, args.specified_ply_path, args.load_mask, args.load_depth, args.load_norm, args.load_midas, args.is_spin)
            else:
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, ply_path=ply_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if ply_path is not None:
                with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # print(f'self.cameras_extent: {self.cameras_extent}')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            print('Generating Spiral Cameras...')
            # 1. get all poses 
            c2ws_all = []
            for ii in range(len(self.train_cameras[resolution_scale])):
                c2ws_all += [self.train_cameras[resolution_scale][ii].world_view_transform.transpose(0,1).inverse()[:3,:4].cpu().numpy()]
                print(self.train_cameras[resolution_scale][ii].image_name)
            c2ws_all = np.stack(c2ws_all) # N, 3, 4

            # # 2. get spiral poses
            # near_fars = np.array([self.train_cameras[resolution_scale][0].znear, self.train_cameras[resolution_scale][0].zfar]).astype(np.float32)
            poses_bounds = np.load(os.path.join(args.source_path, 'poses_bounds.npy'))  # (N_images, 17)
            print()


            camdata = read_cameras_binary(os.path.join(args.source_path, 'sparse/0/cameras.bin'))
            intr = camdata[1]
            if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
                cx = intr.params[1]
                cy = intr.params[2]
            elif intr.model=="PINHOLE":
                cx = intr.params[2]
                cy = intr.params[3]
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported! %s"%intr.model

            cx = (cx - intr.width / 2) / intr.width * 2
            cy = (cy - intr.height / 2) / intr.height * 2
            
            imdata = read_images_binary(os.path.join(args.source_path, 'sparse/0/images.bin'))
            perm = np.argsort([imdata[k].name for k in imdata])

            print(os.path.join(args.source_path, 'poses_bounds.npy'))
            print(poses_bounds.shape, poses_bounds.dtype, poses_bounds.min(), poses_bounds.max())

            

            near_fars = poses_bounds[40:][:, -2:]
            # near_fars = np.array([1.3333333333333335, 6.865135149182821]).astype(np.float32)
            h, w = self.train_cameras[resolution_scale][0].image_width, self.train_cameras[resolution_scale][0].image_height
            FovX, FovY = self.train_cameras[resolution_scale][0].FoVx, self.train_cameras[resolution_scale][0].FoVy

            spiral_c2ws = self.get_spiral(c2ws_all, near_fars, rads_scale=0.4) # n,4,4

            # self.spiral_cameras[resolution_scale] = cameraList_from_camInfos_spiral(spiral_c2ws, h, w, FovX, FovY, resolution_scale, args)
            self.spiral_cameras[resolution_scale] = cameraList_from_camInfos_spiral(spiral_c2ws, h, w, FovX, FovY, cx, cy, resolution_scale, args)


        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.pretrained_model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.pretrained_model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "checkpoint.pth"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def render_path_spiral(self, c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
        render_poses = []
        rads = np.array(list(rads) + [1.])

        for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
            c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
            z = self.normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
            render_poses.append(self.viewmatrix(z, up, c))
        return render_poses

    # get_spiral(self.poses, self.near_fars, N_views=N_views)
    def get_spiral(self, c2ws_all, near_fars, rads_scale=1.0, N_views=120):
        # center pose
        c2w = self.average_poses(c2ws_all)

        # Get average pose
        up = self.normalize(c2ws_all[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        dt = 0.75
        close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
        focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        # focal = -1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

        # Get radii for spiral path
        zdelta = near_fars.min() * .2
        tt = c2ws_all[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
        render_poses = self.render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
        return np.stack(render_poses)
    
    
    def normalize(self, v):
        """Normalize a vector."""
        return v / np.linalg.norm(v)


    def average_poses(self, poses):
        """
        Calculate the average pose, which is then used to center all poses
        using @center_poses. Its computation is as follows:
        1. Compute the center: the average of pose centers.
        2. Compute the z axis: the normalized average z axis.
        3. Compute axis y': the average y axis.
        4. Compute x' = y' cross product z, then normalize it as the x axis.
        5. Compute the y axis: z cross product x.

        Note that at step 3, we cannot directly use y' as y axis since it's
        not necessarily orthogonal to z axis. We need to pass from x to y.
        Inputs:
            poses: (N_images, 3, 4)
        Outputs:
            pose_avg: (3, 4) the average pose
        """
        # 1. Compute the center
        center = poses[..., 3].mean(0)  # (3)

        # 2. Compute the z axis
        z = self.normalize(poses[..., 2].mean(0))  # (3)

        # 3. Compute axis y' (no need to normalize as it's not the final output)
        y_ = poses[..., 1].mean(0)  # (3)

        # 4. Compute the x axis
        x = self.normalize(np.cross(z, y_))  # (3)

        # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
        y = np.cross(x, z)  # (3)

        pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

        return pose_avg

    def viewmatrix(self, z, up, pos):
        vec2 = self.normalize(z)
        vec1_avg = up
        vec0 = self.normalize(np.cross(vec1_avg, vec2))
        vec1 = self.normalize(np.cross(vec2, vec0))
        m = np.eye(4)
        m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
        return m
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(os.path.join(point_cloud_path, "checkpoint.pth"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getSpiralCameras(self, scale=1.0):
        return self.spiral_cameras[scale]
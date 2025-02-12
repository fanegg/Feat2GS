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
# from lietorch import SO3, SE3, Sim3, LieGroupParameter
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial.transform import Rotation as R
from utils.pose_utils import rotation2quad, get_tensor_from_camera
from utils.graphics_utils import getWorld2View2

import torch.nn.functional as F

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.

    Parameters:
    - quaternion: A tensor of shape (..., 4) representing quaternions.

    Returns:
    - A tensor of shape (..., 3, 3) representing rotation matrices.
    """
    # Ensure quaternion is of float type for computation
    quaternion = quaternion.float()

    # Normalize the quaternion to unit length
    quaternion = quaternion / quaternion.norm(p=2, dim=-1, keepdim=True)

    # Extract components
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]

    # Compute rotation matrix components
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    # Assemble the rotation matrix
    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz),     2 * (xy - zw),     2 * (xz + yw)], dim=-1),
        torch.stack([    2 * (xy + zw), 1 - 2 * (xx + zz),     2 * (yz - xw)], dim=-1),
        torch.stack([    2 * (xz - yw),     2 * (yz + xw), 1 - 2 * (xx + yy)], dim=-1)
    ], dim=-2)

    return R


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

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        # self.active_sh_degree = 0
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.param_init = {}
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.P,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale,
        self.P) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    def compute_relative_world_to_camera(self, R1, t1, R2, t2):
        # Create a row of zeros with a one at the end, for homogeneous coordinates
        zero_row = np.array([[0, 0, 0, 1]], dtype=np.float32)

        # Compute the inverse of the first extrinsic matrix
        E1_inv = np.hstack([R1.T, -R1.T @ t1.reshape(-1, 1)])  # Transpose and reshape for correct dimensions
        E1_inv = np.vstack([E1_inv, zero_row])  # Append the zero_row to make it a 4x4 matrix

        # Compute the second extrinsic matrix
        E2 = np.hstack([R2, -R2 @ t2.reshape(-1, 1)])  # No need to transpose R2
        E2 = np.vstack([E2, zero_row])  # Append the zero_row to make it a 4x4 matrix

        # Compute the relative transformation
        E_rel = E2 @ E1_inv

        return E_rel

    def init_RT_seq(self, cam_list):
        poses =[]
        for cam in cam_list[1.0]:
            p = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1)) # R T -> quat t
            poses.append(p)
        poses = torch.stack(poses)
        self.P = poses.cuda().requires_grad_(True)
        # poses_ = torch.randn(poses.detach().clone().shape, device='cuda')
        # self.P = poses_.cuda().requires_grad_(True)
        self.param_init['pose'] = poses.detach().clone()

    def get_RT(self, idx):
        pose = self.P[idx]
        return pose

    def get_RT_test(self, idx):
        pose = self.test_P[idx]
        return pose

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.param_init.update({
            'xyz': fused_point_cloud.detach().clone(),
            'f_dc':  features[:,:,0:1].transpose(1, 2).contiguous().detach().clone(),
            'f_rest': features[:,:,1:].transpose(1, 2).contiguous().detach().clone(),
            'opacity': opacities.detach().clone(),
            'scaling': scales.detach().clone(),
            'rotation': rots.detach().clone(),
        })
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        l_cam = [{'params': [self.P],'lr': training_args.rotation_lr*0.1, "name": "pose"},]
        # l_cam = [{'params': [self.P],'lr': training_args.rotation_lr, "name": "pose"},]

        l += l_cam

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.cam_scheduler_args = get_expon_lr_func(
                                                    # lr_init=0,
                                                    # lr_final=0,
                                                    lr_init=training_args.rotation_lr*0.1,
                                                    lr_final=training_args.rotation_lr*0.001,
                                                    # lr_init=training_args.position_lr_init*self.spatial_lr_scale*10,
                                                    # lr_final=training_args.position_lr_final*self.spatial_lr_scale*10,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=1000)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pose":
                lr = self.cam_scheduler_args(iteration)
                # print("pose learning rate", iteration, lr)
                param_group['lr'] = lr
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
        # return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # breakpoint()
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # self.densify_and_clone(grads, max_grad, extent)
        # self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


class Feat2GaussianModel(GaussianModel):

    def __init__(self, sh_degree : int, feat_dim : int, gs_params_group : dict, noise_std=0):
        super().__init__(sh_degree)
        self.noise_std = noise_std
        self.pc_feat = torch.empty(0)
        self.param_init = {}
        self.feat_dim = feat_dim
        self.gs_params_group = gs_params_group
        self.active_sh_degree = sh_degree
        self.sh_coeffs = ((sh_degree + 1) ** 2) * 3-3
        net_width = feat_dim
        out_dim = {'xyz': 3, 'scaling': 3, 'rotation': 4, 'opacity': 1, 'f_dc': 3, 'f_rest': self.sh_coeffs}
        for key in gs_params_group.get('head', []):
            setattr(self, f'head_{key}', conditionalWarp(layers=[feat_dim, net_width, out_dim[key]], skip=[]).cuda())

        self.param_key = {
            'xyz': '_xyz', 
            'scaling': '_scaling', 
            'rotation': '_rotation', 
            'opacity': '_opacity', 
            'f_dc': '_features_dc', 
            'f_rest': '_features_rest',
            'pc_feat': 'pc_feat',
            }
        
        # ## FOR DEBUGGING
        # self.head_xyz = conditionalWarp(layers=[self.feat_dim, net_width, 3], skip=[]).cuda()
        # self.head_scaling = conditionalWarp(layers=[self.feat_dim, net_width, 3], skip=[]).cuda()
        # self.head_rotation = conditionalWarp(layers=[self.feat_dim, net_width, 4], skip=[]).cuda()
        # self.head_opacity = conditionalWarp(layers=[self.feat_dim, net_width, 1], skip=[]).cuda()
        # self.head_f_dc = conditionalWarp(layers=[feat_dim, net_width, 3], skip=[]).cuda()
        # self.head_f_rest = conditionalWarp(layers=[feat_dim, net_width, self.sh_coeffs], skip=[]).cuda()

    def capture(self):
        head_state_dicts = {f'head_{key}': getattr(self, f'head_{key}').state_dict() for key in self.gs_params_group.get('head', [])}
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.P,
            head_state_dicts
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale,
        self.P,
        head_state_dicts
        ) = model_args
        
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

        for key, state_dict in head_state_dicts.items():
            getattr(self, key).load_state_dict(state_dict)

    def inference(self):   
        feat_in = self.pc_feat
        for key in self.gs_params_group.get('head', []):

            if key == 'f_dc':
                self._features_dc = getattr(self, f'head_{key}')(feat_in, self.param_init[key].view(-1, 3)).reshape(-1, 1, 3)
            elif key == 'f_rest':
                self._features_rest = getattr(self, f'head_{key}')(feat_in.detach(), self.param_init[key].view(-1, self.sh_coeffs)).reshape(-1, self.sh_coeffs // 3, 3)
            else:
                setattr(self, f'_{key}', getattr(self, f'head_{key}')(feat_in, self.param_init[key]))

            # if key == 'f_dc':
            #     self._features_dc = getattr(self, f'head_{key}')(feat_in, self.param_init[key].view(-1, 3)).reshape(-1, 1, 3)
            #     self._features_dc += self.param_init[key].view(-1, 1, 3).mean(dim=0, keepdim=True)
            # elif key == 'f_rest':
            #     self._features_rest = getattr(self, f'head_{key}')(feat_in.detach(), self.param_init[key].view(-1, self.sh_coeffs)).reshape(-1, self.sh_coeffs // 3, 3)
            #     self._features_rest += self.param_init[key].view(-1, self.sh_coeffs // 3, 3).mean(dim=0, keepdim=True)
            # else:
            #     pred = getattr(self, f'head_{key}')(feat_in, self.param_init[key])
            #     setattr(self, f'_{key}', pred + self.param_init[key].mean(dim=0, keepdim=True))
   
        # ## FOR DEBUGGING
        # self._xyz = self.head_xyz(pred, self.param_init['xyz'])
        # self._opacity = self.head_opacity(pred, self.param_init['opacity'])
        # self._scaling = self.head_scaling(pred, self.param_init['scaling'])
        # self._rotation = self.head_rotation(pred, self.param_init['rotation'])
        # self._features_dc = self.head_f_dc(pred, self.param_init['f_dc'].view(-1,3)).reshape(-1, 1, 3)
        # self._features_rest = self.head_f_rest(pred, self.param_init['f_rest'].view(-1,self.sh_coeffs)).reshape(-1, self.sh_coeffs//3, 3)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_point_feat = torch.tensor(np.asarray(pcd.features)).float().cuda()    # get features from .PLY file
        assert fused_point_feat.shape[-1] == self.feat_dim, f"Expected feature dimension {self.feat_dim}, but got {fused_point_feat.shape[-1]}"
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.pc_feat = fused_point_feat#.requires_grad_(True)

        # fused_point_feat = torch.randn_like(fused_point_feat)
        # self.pc_feat = fused_point_feat.requires_grad_(True)

        self.gt_xyz = fused_point_cloud.clone()
        if self.noise_std != 0:
            self.noise_std /= 1000.0
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)    
            noise = torch.randn_like(fused_point_cloud) * self.noise_std
            fused_point_cloud += noise
            # fused_point_cloud = noise + fused_point_cloud.mean(dim=0, keepdim=True)
            # fused_point_cloud = torch.zeros_like(fused_point_cloud) + fused_point_cloud.mean(dim=0, keepdim=True)

        param_init = {
            'xyz': fused_point_cloud,
            'scaling': scales,
            'rotation': rots,
            'opacity': opacities,
            'f_dc': features[:, :, 0:1].transpose(1, 2).contiguous(),
            'f_rest': features[:, :, 1:].transpose(1, 2).contiguous(),
            'pc_feat': fused_point_feat,
        }

        for key in self.gs_params_group.get('opt', []):
            setattr(self, self.param_key[key], nn.Parameter(param_init[key].requires_grad_(True)))

        self.param_init.update({key: value.detach().clone() for key, value in param_init.items()})
        
        # ## FOR DEBUGGING
        # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        # self.param_init.update({
        #     'xyz': fused_point_cloud.detach().clone(),
        #     'f_dc':  features[:,:,0:1].transpose(1, 2).contiguous().detach().clone(),
        #     'f_rest': features[:,:,1:].transpose(1, 2).contiguous().detach().clone(),
        #     'opacity': opacities.detach().clone(),
        #     'scaling': scales.detach().clone(),
        #     'rotation': rots.detach().clone(),
        #     'pc_feat':fused_point_feat.detach().clone(),
        # })

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.param_lr = {
            "xyz": training_args.position_lr_init * self.spatial_lr_scale,
            "f_dc": training_args.feature_lr,
            "f_rest": training_args.feature_sh_lr,
            "opacity": training_args.opacity_lr,
            "scaling": training_args.scaling_lr,
            "rotation": training_args.rotation_lr
        }

        warm_start_lr = 0.01
        l = []
        for key in self.gs_params_group.get('head', []):
            l.append({
                'params': getattr(self, f'head_{key}').parameters(),
                'lr': warm_start_lr,
                'name': key
            })

        for key in self.gs_params_group.get('opt', []):
            l.append({
                'params': [getattr(self, self.param_key[key])], 
                'lr': warm_start_lr,
                'name': key
            })

        # ## FOR DEBUGGING
        # l += [
        #     {'params': self.head_f_dc.parameters(), 'lr': warm_start_lr, "name": "warm_start_f_dc"},
        #     {'params': self.head_f_rest.parameters(), 'lr': warm_start_lr, "name": "warm_start_f_rest"},
        # ]

        # l = [
        #     {'params': self.head_xyz.parameters(), 'lr': warm_start_lr, "name": "xyz"},
        #     # {'params': [self._xyz], 'lr': warm_start_lr, "name": "xyz"},
        #     {'params': self.head_scaling.parameters(), 'lr': warm_start_lr, "name": "scaling"},
        #     # {'params': [self._scaling], 'lr': warm_start_lr, "name": "scaling"},
        #     {'params': self.head_rotation.parameters(), 'lr': warm_start_lr, "name": "rotation"},
        #     # {'params': [self._rotation], 'lr': warm_start_lr, "name": "rotation"},
        #     {'params': self.head_opacity.parameters(), 'lr': warm_start_lr, "name": "opacity"},
        #     # {'params': [self._opacity], 'lr': warm_start_lr, "name": "opacity"},
        #     # {'params': self.head_f_dc.parameters(), 'lr': warm_start_lr, "name": "f_dc"},
        #     {'params': [self._features_dc], 'lr': warm_start_lr, "name": "f_dc"},
        #     # {'params': self.head_f_rest.parameters(), 'lr': warm_start_lr, "name": "f_rest"},
        #     {'params': [self._features_rest], 'lr': warm_start_lr, "name": "f_rest"},
        #     # {'params': [self.pc_feat], 'lr': warm_start_lr, "name": "feat"},
        # ]

        l_cam = [{'params': [self.P],'lr': training_args.pose_lr_init, "name": "pose"},]

        l += l_cam

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.cam_scheduler_args = get_expon_lr_func(lr_init=training_args.pose_lr_init,
                                                    lr_final=training_args.pose_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=1000)
        
        self.warm_start_scheduler_args = get_expon_lr_func(lr_init=warm_start_lr,
                                                           lr_final=warm_start_lr*0.01,
                                                            max_steps=1000)

    def setup_rendering_learning_rate(self, ):
        ''' Setup learning rate scheduling'''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in self.param_lr:
                param_group['lr'] = self.param_lr[param_group["name"]]
            # elif param_group["name"] == "feat":
            #     param_group['lr'] = 1e-6

    def update_warm_start_learning_rate(self, iteration):
        ''' Warm start learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            lr = self.warm_start_scheduler_args(iteration)
            param_group['lr'] = lr

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pose":
                lr = self.cam_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
        # return lr

class conditionalWarp(torch.nn.Module):
    def __init__(self, layers, skip, skip_dim=None, res=[], freq=None, zero_init=False):
        super().__init__()
        self.skip = skip
        self.res = res
        self.freq = freq
        self.mlp_warp = torch.nn.ModuleList()
        L = self.get_layer_dims(layers)
        for li,(k_in,k_out) in enumerate(L):
            if li in self.skip: k_in += layers[-1] if skip_dim is None else skip_dim
            linear = torch.nn.Linear(k_in,k_out)

            # Init network output as 0
            if zero_init:
                if li == (len(L) - 1):
                    torch.nn.init.constant_(linear.weight, 0)
                    torch.nn.init.constant_(linear.bias, 0)

            self.mlp_warp.append(linear)

    def get_layer_dims(self, layers):
        # return a list of tuples (k_in,k_out)
        return list(zip(layers[:-1],layers[1:]))
    
    def positional_encoding(self, input): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(self.freq, dtype=torch.float32,device=input.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc

    def forward(self, feat_in, color):
        if self.freq != None:
            feat_in = torch.cat([feat_in, self.positional_encoding(feat_in)], dim=-1)
        feat = feat_in
        for li,layer in enumerate(self.mlp_warp):
            if li in self.skip: feat = torch.cat([feat, color],dim=-1)
            if li in self.res: feat = feat + feat_in
            feat = layer(feat)
            if li!=len(self.mlp_warp)-1:
                feat = nn.functional.relu(feat)
        warp = feat
        return warp
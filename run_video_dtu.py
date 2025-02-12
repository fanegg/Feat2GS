#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import math
import copy
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_gsplat
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.pose_utils import get_tensor_from_camera
import shutil
import numpy as np
import imageio.v3 as iio
from utils.graphics_utils import resize_render, make_video_divisble
from utils.camera_utils import visualizer
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid
from utils.image_utils import accuracy_torch, completion_torch, accuracy_per_point, completion_per_point
import roma

from utils.trajectories import (
    get_arc_w2cs,
    get_avg_w2c,
    get_lemniscate_w2cs,
    get_spiral_w2cs,
    get_wander_w2cs,
    get_lookat,
)

from utils.camera_utils import generate_interpolated_path, generate_ellipse_path
from utils.camera_traj_config import trajectory_configs


def copy_gaussian_model(source_model):
    new_model = GaussianModel(source_model.max_sh_degree)
    new_model._xyz = torch.nn.Parameter(source_model._xyz.clone().detach())
    new_model._features_dc = torch.nn.Parameter(source_model._features_dc.clone().detach())
    new_model._features_rest = torch.nn.Parameter(source_model._features_rest.clone().detach())
    new_model._scaling = torch.nn.Parameter(source_model._scaling.clone().detach())
    new_model._rotation = torch.nn.Parameter(source_model._rotation.clone().detach())
    new_model._opacity = torch.nn.Parameter(source_model._opacity.clone().detach())
    new_model.max_sh_degree = source_model.max_sh_degree
    return new_model

def make_point_cloud_gaussian(gaussians, features):
    gaussians._features_dc = torch.nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    gaussians._features_rest = torch.nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    gaussians._scaling = torch.nn.Parameter(torch.log(torch.sqrt(0.0000001 * torch.ones_like(gaussians._scaling))).requires_grad_(True))
    # gaussians._scaling = torch.nn.Parameter(torch.log(torch.sqrt(0.00000001 * torch.ones_like(gaussians._scaling))).requires_grad_(True))

    rots = torch.zeros_like(gaussians._rotation)
    rots[:, 0] = 1
    gaussians._rotation = torch.nn.Parameter(rots.requires_grad_(True))
    gaussians._opacity = torch.nn.Parameter(inverse_sigmoid(0.9 * torch.ones_like(gaussians._opacity)).requires_grad_(True))


def error_to_color(error, vmin=None, vmax=None):
    if vmin is None:
        vmin = error.min()
    if vmax is None:
        vmax = error.max()
    
    # Normalize to [0,1] range
    error = torch.clamp(error, vmin, vmax)
    normalized = (error - vmin) / (vmax - vmin)
    
    # Use jet colormap
    colors = torch.zeros((error.shape[0], 3), device=error.device)
    
    # 4 intervals of jet colormap
    colors[:, 0] = torch.clamp(torch.minimum(4 * normalized - 1.5, -4 * normalized + 4.5), 0, 1)
    colors[:, 1] = torch.clamp(torch.minimum(4 * normalized - 0.5, -4 * normalized + 3.5), 0, 1)
    colors[:, 2] = torch.clamp(torch.minimum(4 * normalized + 0.5, -4 * normalized + 2.5), 0, 1)
    
    return colors


def save_interpolated_pose(model_path, iter, n_views):

    org_pose = np.load(model_path + f"pose/pose_{iter}.npy")
    # visualizer(org_pose, ["green" for _ in org_pose], model_path + "pose/poses_optimized.png")
    n_interp = int(10 * 30 / n_views)  # 10second, fps=30
    all_inter_pose = []
    for i in range(n_views-1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.array(all_inter_pose).reshape(-1, 3, 4)

    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    return inter_pose


def save_ellipse_pose(model_path, iter, n_views):

    org_pose = np.load(model_path + f"pose/pose_{iter}.npy")
    # visualizer(org_pose, ["green" for _ in org_pose], model_path + "pose/poses_optimized.png")
    n_interp = int(10 * 30 / n_views) * (n_views-1)  # 10second, fps=30
    all_inter_pose = generate_ellipse_path(org_pose, n_interp)

    inter_pose_list = []
    for p in all_inter_pose:
        c2w = np.eye(4)
        c2w[:3, :4] = p
        inter_pose_list.append(np.linalg.inv(c2w))
    inter_pose = np.stack(inter_pose_list, 0)

    return inter_pose

def save_traj_pose(dataset, iter, args):

    traj_up = trajectory_configs.get(args.dataset, {}).get(args.scene, {}).get('up', [-1, 1])  # Use camera space -y axis as up vector
    traj_params = trajectory_configs.get(args.dataset, {}).get(args.scene, {}).get(args.cam_traj, {})
    
    # 1. Get training camera poses and calculate trajectory
    org_pose = np.load(dataset.model_path + f"pose/pose_{iter}.npy")
    train_w2cs = torch.from_numpy(org_pose).cuda()
    
    # Calculate reference camera pose
    avg_w2c = get_avg_w2c(train_w2cs)
    train_c2ws = torch.linalg.inv(train_w2cs)
    lookat = get_lookat(train_c2ws[:, :3, -1], train_c2ws[:, :3, 2])
    # up = torch.tensor([0.0, 0.0, 1.0], device="cuda")
    avg_c2w = torch.linalg.inv(avg_w2c)
    up = traj_up[0] * (avg_c2w[:3, traj_up[1]])
    # up = traj_up[0] * (avg_c2w[:3, 0]+avg_c2w[:3, 1])/2

    # Temporarily load a camera to get intrinsic parameters
    tmp_args = copy.deepcopy(args)
    tmp_args.get_video = False
    tmp_dataset = copy.deepcopy(dataset)
    tmp_dataset.eval = False
    with torch.no_grad():
        temp_gaussians = GaussianModel(dataset.sh_degree)    
        temp_scene = Scene(tmp_dataset, temp_gaussians, load_iteration=iter, opt=tmp_args, shuffle=False)

    view = temp_scene.getTrainCameras()[0]
    tanfovx = math.tan(view.FoVx * 0.5)
    tanfovy = math.tan(view.FoVy * 0.5)
    focal_length_x = view.image_width / (2 * tanfovx)
    focal_length_y = view.image_height / (2 * tanfovy)

    K = torch.tensor([[focal_length_x, 0, view.image_width/2],
                     [0, focal_length_y, view.image_height/2],
                     [0, 0, 1]], device="cuda")
    img_wh = (view.image_width, view.image_height)

    del temp_scene  # Release temporary scene
    del temp_gaussians  # Release temporary gaussians

    # Calculate bounding sphere radius
    rc_train_c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(avg_w2c), train_c2ws)
    rc_pos = rc_train_c2ws[:, :3, -1]
    rads = (rc_pos.amax(0) - rc_pos.amin(0)) * 1.25

    num_frames = int(10 * 30 / args.n_views) * (args.n_views-1)

    # Generate camera poses based on trajectory type
    if args.cam_traj == 'arc':
        w2cs = get_arc_w2cs(
            ref_w2c=avg_w2c,
            lookat=lookat,
            up=up,
            focal_length=K[0, 0].item(),
            rads=rads,
            num_frames=num_frames,
            degree=traj_params.get('degree', 180.0)
        )
    elif args.cam_traj == 'spiral':
        w2cs = get_spiral_w2cs(
            ref_w2c=avg_w2c,
            lookat=lookat,
            up=up,
            focal_length=K[0, 0].item(),
            rads=rads,
            num_frames=num_frames,
            zrate=traj_params.get('zrate', 0.5),
            rots=traj_params.get('rots', 1)
        )
    elif args.cam_traj == 'lemniscate':
        w2cs = get_lemniscate_w2cs(
            ref_w2c=avg_w2c,
            lookat=lookat,
            up=up,
            focal_length=K[0, 0].item(),
            rads=rads,
            num_frames=num_frames,
            degree=traj_params.get('degree', 45.0)
        )
    elif args.cam_traj == 'wander':
        w2cs = get_wander_w2cs(
            ref_w2c=avg_w2c,
            focal_length=K[0, 0].item(),
            num_frames=num_frames,
            max_disp=traj_params.get('max_disp', 48.0)
        )
    else:
        raise ValueError(f"Unknown camera trajectory: {args.cam_traj}")

    return w2cs.cpu().numpy()

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, args):
    if args.cam_traj in ['interpolated', 'ellipse']:
        w2cs = globals().get(f'save_{args.cam_traj}_pose')(dataset.model_path, iteration, args.n_views)
    else:
        w2cs = save_traj_pose(dataset, iteration, args)

    # visualizer(org_pose, ["green" for _ in org_pose], dataset.model_path + f"pose/poses_optimized.png")
    visualizer(w2cs, ["blue" for _ in w2cs], dataset.model_path + f"pose/poses_{args.cam_traj}.png")
    np.save(dataset.model_path + f"pose/pose_{args.cam_traj}.npy", w2cs)


    # 2. Load model and scene
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)

        # load ground truth gaussians
        # gaussians.create_from_pcd(scene_info.point_cloud, scene_info.nerf_normalization["radius"])
        scene_info = sceneLoadTypeCallbacks["Colmap"](dataset.source_path, dataset.images, dataset.eval, dataset, args)
        ground_truth_point_cloud = torch.tensor(np.asarray(scene_info.point_cloud.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(scene_info.point_cloud.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (dataset.sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # 3D metrics
        # Rigid registration
        R, T, s = roma.rigid_points_registration(
            gaussians._xyz.reshape(-1, 3), ground_truth_point_cloud.reshape(-1, 3), weights=None, compute_scaling=True)
        # Build transformation matrix
        transform_matrix = torch.eye(4, device="cuda")
        transform_matrix[:3, :3] = R * s
        transform_matrix[:3, 3] = T
        # Apply transformation
        pred_xyz_aligned = (transform_matrix[:3, :3] @ gaussians._xyz.T + transform_matrix[:3, 3:]).T
        # acc, _ = accuracy_torch(ground_truth_point_cloud, pred_xyz_aligned)
        # comp, _ = completion_torch(ground_truth_point_cloud, pred_xyz_aligned)
        # dist = torch.mean(torch.norm(ground_truth_point_cloud - pred_xyz_aligned, dim=-1))
        acc = accuracy_per_point(ground_truth_point_cloud, pred_xyz_aligned, batch_size=1000)
        comp = completion_per_point(ground_truth_point_cloud, pred_xyz_aligned, batch_size=1000)
        dist = torch.norm(ground_truth_point_cloud - pred_xyz_aligned, dim=-1)

        acc_colors = error_to_color(acc, vmin=0, vmax=10.0/1000.0)
        comp_colors = error_to_color(comp, vmin=0, vmax=10.0/1000.0)
        dist_colors = error_to_color(dist, vmin=0, vmax=10.0/1000.0)
        # acc_colors = error_to_color(acc)
        # comp_colors = error_to_color(comp)
        # dist_colors = error_to_color(dist)
        acc_colors = RGB2SH(acc_colors)
        comp_colors = RGB2SH(comp_colors)
        dist_colors = RGB2SH(dist_colors)

        acc_features = torch.zeros((acc_colors.shape[0], 3, (dataset.sh_degree + 1) ** 2)).float().cuda()
        acc_features[:, :3, 0 ] = acc_colors
        acc_features[:, 3:, 1:] = 0.0
        comp_features = torch.zeros((comp_colors.shape[0], 3, (dataset.sh_degree + 1) ** 2)).float().cuda()
        comp_features[:, :3, 0 ] = comp_colors
        comp_features[:, 3:, 1:] = 0.0
        dist_features = torch.zeros((dist_colors.shape[0], 3, (dataset.sh_degree + 1) ** 2)).float().cuda()
        dist_features[:, :3, 0 ] = dist_colors
        dist_features[:, 3:, 1:] = 0.0

        # Rigid registration
        R, T, s = roma.rigid_points_registration(
            ground_truth_point_cloud.reshape(-1, 3), gaussians._xyz.reshape(-1, 3), weights=None, compute_scaling=True)
        # Build transformation matrix
        transform_matrix = torch.eye(4, device="cuda") 
        transform_matrix[:3, :3] = R * s
        transform_matrix[:3, 3] = T
        # Apply transformation
        ground_truth_point_cloud_aligned = (transform_matrix[:3, :3] @ ground_truth_point_cloud.T + transform_matrix[:3, 3:]).T

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)    
        noise = torch.randn_like(ground_truth_point_cloud) * (10.0 / 1000.0)
        noised_point_cloud = ground_truth_point_cloud + noise
        noised_point_cloud_aligned = (transform_matrix[:3, :3] @ noised_point_cloud.T + transform_matrix[:3, 3:]).T

        # ground truth gaussians
        ground_truth = copy_gaussian_model(gaussians)
        ground_truth._xyz = torch.nn.Parameter(ground_truth_point_cloud_aligned.requires_grad_(True))
        make_point_cloud_gaussian(ground_truth, features)

        ground_truth_comp = copy_gaussian_model(gaussians)
        ground_truth_comp._xyz = torch.nn.Parameter(ground_truth_point_cloud_aligned.requires_grad_(True))
        make_point_cloud_gaussian(ground_truth_comp, comp_features)

        # noised point cloud gaussians
        noised_ground_truth = copy_gaussian_model(gaussians)
        noised_ground_truth._xyz = torch.nn.Parameter(noised_point_cloud_aligned.requires_grad_(True))
        make_point_cloud_gaussian(noised_ground_truth, features)

        # point cloud gaussians
        gaussians_point_cloud = copy_gaussian_model(gaussians)
        make_point_cloud_gaussian(gaussians_point_cloud, features)
        gaussians_point_cloud_acc = copy_gaussian_model(gaussians)
        make_point_cloud_gaussian(gaussians_point_cloud_acc, acc_features)
        gaussians_point_cloud_dist = copy_gaussian_model(gaussians)
        make_point_cloud_gaussian(gaussians_point_cloud_dist, dist_features)

        # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 3. Render
    # render_path = os.path.join(dataset.model_path, args.cam_traj, f"ours_{iteration}", "renders")
    # if os.path.exists(render_path):
    #     shutil.rmtree(render_path)
    # makedirs(render_path, exist_ok=True)

    video,gt_video,noised_gt_video,point_cloud_video,point_cloud_acc_video,point_cloud_dist_video,gt_comp_video = [],[],[],[],[],[],[]
    for idx, w2c in enumerate(tqdm(w2cs, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(w2c.transpose(0, 1))
        view = scene.getTrainCameras()[0]  # Use parameters from the first camera as template
        if args.resize:
            view = resize_render(view)
            
        rendering = render_gsplat(
            view, gaussians, pipeline, background, camera_pose=camera_pose
        )["render"]
        
        # # Save single frame image
        # torchvision.utils.save_image(
        #     rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        # )
        
        ground_truth_rendering = render_gsplat(
            view, ground_truth, pipeline, background, camera_pose=camera_pose
        )["render"]
        
        # torchvision.utils.save_image(
        #     ground_truth_rendering, os.path.join(render_path, "{0:05d}".format(idx) + "_gt.png")
        # )

        noised_ground_truth_rendering = render_gsplat(
            view, noised_ground_truth, pipeline, background, camera_pose=camera_pose
        )["render"]
        
        # torchvision.utils.save_image(
        #     noised_ground_truth_rendering, os.path.join(render_path, "{0:05d}".format(idx) + "_noised_gt.png")
        # )

        gaussians_point_cloud_rendering = render_gsplat(
            view, gaussians_point_cloud, pipeline, background, camera_pose=camera_pose
        )["render"]
        
        # torchvision.utils.save_image(
        #     gaussians_point_cloud_rendering, os.path.join(render_path, "{0:05d}".format(idx) + "_point_cloud2.png")
        # )

        gaussians_point_cloud_rendering_acc = render_gsplat(
            view, gaussians_point_cloud_acc, pipeline, background, camera_pose=camera_pose
        )["render"]
        # torchvision.utils.save_image(
        #     gaussians_point_cloud_rendering_acc, os.path.join(render_path, "{0:05d}".format(idx) + "_point_cloud_acc.png")
        # )
        ground_truth_rendering_comp = render_gsplat(
            view, ground_truth_comp, pipeline, background, camera_pose=camera_pose
        )["render"]
        # torchvision.utils.save_image(
        #     ground_truth_rendering_comp, os.path.join(render_path, "{0:05d}".format(idx) + "_gt_comp.png")
        # )
        gaussians_point_cloud_rendering_dist = render_gsplat(
            view, gaussians_point_cloud_dist, pipeline, background, camera_pose=camera_pose
        )["render"]
        # torchvision.utils.save_image(
        #     gaussians_point_cloud_rendering_dist, os.path.join(render_path, "{0:05d}".format(idx) + "_point_cloud_dist.png")
        # )
        # Add to video list
        # img = (rendering.detach().cpu().numpy() * 255.0).astype(np.uint8)
        img = (torch.clamp(rendering, 0, 1).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        video.append(img)
        gt_video.append((torch.clamp(ground_truth_rendering, 0, 1).detach().cpu().numpy() * 255.0).round().astype(np.uint8))
        noised_gt_video.append((torch.clamp(noised_ground_truth_rendering, 0, 1).detach().cpu().numpy() * 255.0).round().astype(np.uint8))
        point_cloud_video.append((torch.clamp(gaussians_point_cloud_rendering, 0, 1).detach().cpu().numpy() * 255.0).round().astype(np.uint8))
        point_cloud_acc_video.append((torch.clamp(gaussians_point_cloud_rendering_acc, 0, 1).detach().cpu().numpy() * 255.0).round().astype(np.uint8))
        point_cloud_dist_video.append((torch.clamp(gaussians_point_cloud_rendering_dist, 0, 1).detach().cpu().numpy() * 255.0).round().astype(np.uint8))
        gt_comp_video.append((torch.clamp(ground_truth_rendering_comp, 0, 1).detach().cpu().numpy() * 255.0).round().astype(np.uint8))

    video = np.stack(video, 0).transpose(0, 2, 3, 1)
    gt_video = np.stack(gt_video, 0).transpose(0, 2, 3, 1)
    noised_gt_video = np.stack(noised_gt_video, 0).transpose(0, 2, 3, 1)
    point_cloud_video = np.stack(point_cloud_video, 0).transpose(0, 2, 3, 1)
    point_cloud_acc_video = np.stack(point_cloud_acc_video, 0).transpose(0, 2, 3, 1)
    point_cloud_dist_video = np.stack(point_cloud_dist_video, 0).transpose(0, 2, 3, 1)
    gt_comp_video = np.stack(gt_comp_video, 0).transpose(0, 2, 3, 1)

    # Save video
    if args.get_video:
        video_dir = os.path.join(dataset.model_path, 'videos')
        os.makedirs(video_dir, exist_ok=True)

        output_video_file = os.path.join(video_dir, f'{args.scene}_{args.n_views}_view_{args.cam_traj}.mp4')
        # iio.imwrite(output_video_file, make_video_divisble(video), fps=30)
        iio.imwrite(
            output_video_file, 
            make_video_divisble(video), 
            fps=30,
            codec='libx264',
            quality=None,
            output_params=[
                '-crf', '28',
                '-preset', 'veryslow',  
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ]
        )
        gt_video_file = os.path.join(video_dir, f'{args.scene}_{args.n_views}_view_{args.cam_traj}_gt.mp4')
        iio.imwrite(gt_video_file, make_video_divisble(gt_video), fps=30,
                    codec='libx264', 
                    quality=None,
                    output_params=[
                        '-crf', '28',
                        '-preset', 'veryslow',  
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart'
                    ]
                    )
        noised_gt_video_file = os.path.join(video_dir, f'{args.scene}_{args.n_views}_view_{args.cam_traj}_noised_gt.mp4')
        iio.imwrite(noised_gt_video_file, make_video_divisble(noised_gt_video), fps=30,
                    codec='libx264', 
                    quality=None,
                    output_params=[
                        '-crf', '28',
                        '-preset', 'veryslow',  
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart'
                    ]
                    )
        point_cloud_video_file = os.path.join(video_dir, f'{args.scene}_{args.n_views}_view_{args.cam_traj}_point_cloud.mp4')
        iio.imwrite(point_cloud_video_file, make_video_divisble(point_cloud_video), fps=30,
                    codec='libx264',
                    quality=None,
                    output_params=[
                        '-crf', '28',
                        '-preset', 'veryslow',  
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart'
                    ]
                    )
        point_cloud_acc_video_file = os.path.join(video_dir, f'{args.scene}_{args.n_views}_view_{args.cam_traj}_point_cloud_acc.mp4')
        iio.imwrite(point_cloud_acc_video_file, make_video_divisble(point_cloud_acc_video), fps=30,
                    codec='libx264',
                    quality=None,
                    output_params=[
                        '-crf', '28',
                        '-preset', 'veryslow',  
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart'
                    ]
                    )

        point_cloud_dist_video_file = os.path.join(video_dir, f'{args.scene}_{args.n_views}_view_{args.cam_traj}_point_cloud_dist.mp4')
        iio.imwrite(point_cloud_dist_video_file, make_video_divisble(point_cloud_dist_video), fps=30,
                    codec='libx264',
                    quality=None,
                    output_params=[
                        '-crf', '28',
                        '-preset', 'veryslow',  
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart'
                    ]
                    )
        gt_comp_video_file = os.path.join(video_dir, f'{args.scene}_{args.n_views}_view_{args.cam_traj}_gt_comp.mp4')
        iio.imwrite(gt_comp_video_file, make_video_divisble(gt_comp_video), fps=30,
                    codec='libx264',
                    quality=None,
                    output_params=[
                        '-crf', '28',
                        '-preset', 'veryslow',  
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart'
                    ]
                    )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--n_views", default=120, type=int)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--cam_traj", default='arc', type=str, 
                       choices=['arc', 'spiral', 'lemniscate', 'wander', 'interpolated', 'ellipse'],
                       help="Camera trajectory type")
    parser.add_argument("--resize", action="store_true", default=True, 
                       help="If True, resize rendering to square")
    parser.add_argument("--feat_type", type=str, nargs='*', default=None, 
                       help="Feature type(s). Multiple types can be specified for combination.")
    parser.add_argument("--method", type=str, default='dust3r', 
                       help="Method of Initialization, e.g., 'dust3r' or 'mast3r'")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args,
    )
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

import matplotlib
matplotlib.use('Agg')

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
from PIL import Image
import torchvision.transforms.functional as tf
from scene.dataset_readers import sceneLoadTypeCallbacks
import numpy as np
from utils.sh_utils import RGB2SH
import roma
from utils.image_utils import accuracy_per_point, completion_per_point
from run_video_dtu import copy_gaussian_model, make_point_cloud_gaussian, error_to_color

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        # rendering = render(
        #     view, gaussians, pipeline, background, camera_pose=camera_pose
        # )["render"]
        rendering = render_gsplat(
            view, gaussians, pipeline, background, camera_pose=camera_pose
        )["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )

def render_set_optimize(model_path, name, iteration, views, gaussians, pipeline, background, msk_scr_path, args, 
                        ground_truth, ground_truth_comp, noised_ground_truth, gaussians_point_cloud, gaussians_point_cloud_acc, gaussians_point_cloud_dist):
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    msks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")
    point_clouds_path = os.path.join(model_path, name, "ours_{}".format(iteration), "point_clouds")
    msk_suffix = os.path.basename(os.listdir(msk_scr_path)[0]).split('.')[-1]

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(msks_path, exist_ok=True)
    makedirs(point_clouds_path, exist_ok=True)
    
    if args.render_depth_normal:
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
        normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")
        makedirs(depth_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)

    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        num_iter = args.optim_test_pose_iter
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))

        camera_tensor_T = camera_pose[-3:].requires_grad_()
        camera_tensor_q = camera_pose[:4].requires_grad_()
        pose_optimizer = torch.optim.Adam(
            [
                {
                    "params": [camera_tensor_T],
                    "lr": 0.0003,
                },
                {
                    "params": [camera_tensor_q],
                    "lr": 0.0001,
                },
            ]
        )


        progress_bar = tqdm(
            range(num_iter), desc=f"Tracking Time Step: {idx}", disable=True
        )

        # Keep track of best pose candidate
        candidate_q = camera_tensor_q.clone().detach()
        candidate_T = camera_tensor_T.clone().detach()
        current_min_loss = float(1e20)
        gt = view.original_image[0:3, :, :]

        mask = Image.open(os.path.join(msk_scr_path, view.image_name + '.' + msk_suffix))
        if view.image_width > 1600:
            scale = view.image_width / 1600
        else:
            scale = 1.
        resolution = (int(view.image_height / scale), int(view.image_width / scale))
        mask = tf.resize(tf.to_tensor(mask), resolution)

        for iteration in range(num_iter):
            # rendering = render(view, gaussians, pipeline, background, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))["render"]
            rendering = render_gsplat(
                view, gaussians, pipeline, background, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T])
            )["render"]
            # loss = torch.abs(gt - rendering).mean()
            loss = torch.abs((gt - rendering) * mask.to(gt.device)).mean()
            if iteration%10==0:
                print(iteration, loss.item())
            loss.backward()

            with torch.no_grad():
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)

                if iteration == 0:
                    initial_loss = loss

                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_q = camera_tensor_q.clone().detach()
                    candidate_T = camera_tensor_T.clone().detach()

                progress_bar.update(1)

        camera_tensor_q = candidate_q
        camera_tensor_T = candidate_T

        progress_bar.close()
        opt_pose = torch.cat([camera_tensor_q, camera_tensor_T])
        print(opt_pose-camera_pose)
        # rendering_opt = render(view, gaussians, pipeline, background, camera_pose=opt_pose)["render"]
        rendering_opt = render_gsplat(view, gaussians, pipeline, background,
                                            camera_pose=opt_pose, 
                                            render_mode="RGB+ED" if args.render_depth_normal else "RGB")["render"]

        if args.render_depth_normal:
            depth_map = rendering_opt[3, :, :]
            # depth_map = torch.log(depth_map+1.0)
            # depth_map = depth_map.clip(0.05, 0.08)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()).clip(0, 1)
            # depth_map = (8*depth_map).clip(0, 1)

            cm = matplotlib.colormaps["Spectral"]
            depth_map = cm(depth_map.detach().cpu(), bytes=False)[..., 0:3]  # value from 0 to 1
            depth_map = torch.from_numpy(depth_map).to('cuda').permute(2, 0, 1)

            torchvision.utils.save_image(
                depth_map, os.path.join(depth_path, "{0:05d}".format(idx) + ".png")
            )


            normal_map = rendering_opt[4:, :, :]
            normal_map = (normal_map * 0.5 + 0.5)
            normal_map = (normal_map - normal_map.min()) / (
                normal_map.max() - normal_map.min()
            )
            torchvision.utils.save_image(
                normal_map, os.path.join(normal_path, "{0:05d}".format(idx) + ".png")
            )

            rendering_opt = rendering_opt[:3, :, :]

        torchvision.utils.save_image(
            rendering_opt, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            mask, os.path.join(msks_path, "{0:05d}".format(idx) + ".png")
        )

        # point cloud
        rendering_opt_gt = render_gsplat(view, ground_truth, pipeline, background,
                                            camera_pose=opt_pose)["render"]
        torchvision.utils.save_image(
            rendering_opt_gt, os.path.join(point_clouds_path, "{0:05d}".format(idx) + "_gt.png")
        )        

        rendering_opt_gt_comp = render_gsplat(view, ground_truth_comp, pipeline, background,
                                            camera_pose=opt_pose)["render"]
        torchvision.utils.save_image(
            rendering_opt_gt_comp, os.path.join(point_clouds_path, "{0:05d}".format(idx) + "_gt_comp.png")
        )

        rendering_opt_pc = render_gsplat(view, gaussians_point_cloud, pipeline, background,
                                            camera_pose=opt_pose)["render"]
        torchvision.utils.save_image(
            rendering_opt_pc, os.path.join(point_clouds_path, "{0:05d}".format(idx) + "_pc.png")
        )

        rendering_opt_noised_gt = render_gsplat(view, noised_ground_truth, pipeline, background,
                                            camera_pose=opt_pose)["render"]
        torchvision.utils.save_image(
            rendering_opt_noised_gt, os.path.join(point_clouds_path, "{0:05d}".format(idx) + "_noised_gt.png")
        )        

        rendering_opt_acc = render_gsplat(view, gaussians_point_cloud_acc, pipeline, background,
                                            camera_pose=opt_pose)["render"]
        torchvision.utils.save_image(
            rendering_opt_acc, os.path.join(point_clouds_path, "{0:05d}".format(idx) + "_acc.png")
        )
        rendering_opt_dist = render_gsplat(view, gaussians_point_cloud_dist, pipeline, background,
                                            camera_pose=opt_pose)["render"]
        torchvision.utils.save_image(
            rendering_opt_dist, os.path.join(point_clouds_path, "{0:05d}".format(idx) + "_dist.png")
        )

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_test: bool,
    args,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

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
        acc = accuracy_per_point(ground_truth_point_cloud, pred_xyz_aligned, batch_size=1000)
        comp = completion_per_point(ground_truth_point_cloud, pred_xyz_aligned, batch_size=1000)
        dist = torch.norm(ground_truth_point_cloud - pred_xyz_aligned, dim=-1)

        acc_colors = error_to_color(acc, vmin=0, vmax=10.0/1000.0)
        comp_colors = error_to_color(comp, vmin=0, vmax=10.0/1000.0)
        dist_colors = error_to_color(dist, vmin=0, vmax=10.0/1000.0)
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


    msk_path = os.path.join(args.source_path, f"test_view/masks")

    if not skip_test:
        render_set_optimize(
            dataset.model_path,
            "test",
            scene.loaded_iter,
            scene.getTestCameras(),
            gaussians,
            pipeline,
            background,
            msk_path,
            args,
            ground_truth,
            ground_truth_comp,
            noised_ground_truth,
            gaussians_point_cloud,
            gaussians_point_cloud_acc,
            gaussians_point_cloud_dist,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--n_views", default=None, type=int)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)
    parser.add_argument("--method", type=str, default='dust3r', help="Method of Initialization, e.g., 'dust3r' or 'mast3r'")
    # parser.add_argument("--feat_type", type=str, default=None, help="Type of feature to use")
    parser.add_argument("--feat_type", type=str, nargs='*', default=None, help="Feature type(s). Multiple types can be specified for combination.")
    parser.add_argument("--render_depth_normal", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_test,
        args,
    )

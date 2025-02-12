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

def render_set_optimize(model_path, name, iteration, views, gaussians, pipeline, background, msk_scr_path, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    msks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")
    msk_suffix = os.path.basename(os.listdir(msk_scr_path)[0]).split('.')[-1]

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(msks_path, exist_ok=True)
    
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
            args
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

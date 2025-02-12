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

import os
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_gsplat, network_gui
import sys
from scene import Scene, Feat2GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, accuracy_torch, completion_torch
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.pose_utils import get_camera_from_tensor

import roma
import json

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
from time import perf_counter

def compute_3d_metrics(pred_xyz, gt_xyz, pred_opacity, thresholds=[0.0]):
    """
    Perform rigid alignment of predicted point cloud and compute 3D evaluation metrics
    
    Args:
        pred_xyz: Predicted point cloud coordinates (tensor)
        gt_xyz: Ground truth point cloud coordinates (tensor)
        
    Returns:
        tuple: (accuracy, accuracy_median, completion, completion_median)
    """
    # Rigid registration
    R, T, s = roma.rigid_points_registration(
        pred_xyz.reshape(-1, 3), gt_xyz.reshape(-1, 3), weights=None, compute_scaling=True)
    
    # Build transformation matrix
    transform_matrix = torch.eye(4, device="cuda") 
    transform_matrix[:3, :3] = R * s
    transform_matrix[:3, 3] = T
    
    # Apply transformation
    pred_xyz_aligned = (transform_matrix[:3, :3] @ pred_xyz.T + transform_matrix[:3, 3:]).T
    
    metrics = {}
    
    for threshold in thresholds:
        mask = pred_opacity.squeeze() >= threshold
        pred_xyz_masked = pred_xyz_aligned[mask]
        gt_xyz_masked = gt_xyz[mask]
            
        acc, acc_med = accuracy_torch(gt_xyz, pred_xyz_masked)
        comp, comp_med = completion_torch(gt_xyz, pred_xyz_masked)
        dist = torch.mean(torch.norm(gt_xyz_masked - pred_xyz_masked, dim=-1))
        dist_med = torch.median(torch.norm(gt_xyz_masked - pred_xyz_masked, dim=-1))
        
        metrics[threshold] = {
            'Accuracy': acc,
            'Acc_med': acc_med,
            'Completion': comp, 
            'Comp_med': comp_med,
            'Distance': dist,
            'Dist_med': dist_med
        }
    return metrics

def save_pose(path, quat_pose, train_cams, llffhold=2):
    output_poses=[]
    index_colmap = [cam.colmap_id for cam in train_cams]
    for quat_t in quat_pose:
        w2c = get_camera_from_tensor(quat_t)
        output_poses.append(w2c)
    colmap_poses = []
    for i in range(len(index_colmap)):
        ind = index_colmap.index(i+1)
        bb=output_poses[ind]
        bb = bb#.inverse()
        colmap_poses.append(bb)
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt.iterations)
    feat_type = '-'.join(args.feat_type)
    feat_dim = args.feat_dim if feat_type not in ['iuv', 'iuvrgb'] else dataset.feat_default_dim[feat_type]
    gs_params_group = dataset.gs_params_group[args.model]
    gaussians = Feat2GaussianModel(dataset.sh_degree, feat_dim, gs_params_group, noise_std=args.noise_std)
    scene = Scene(dataset, gaussians, opt=args, shuffle=True)                                                                      
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    train_cams_init = scene.getTrainCameras().copy()
    os.makedirs(scene.model_path + 'pose', exist_ok=True)
    save_pose(scene.model_path + 'pose' + "/pose_org.npy", gaussians.P, train_cams_init)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    warm_iter = 1000

    metrics_path = os.path.join(scene.model_path, "3d_metrics.json")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    start = perf_counter()
    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        if iteration > warm_iter:
            if iteration == warm_iter+1:
                gaussians.pc_feat.requires_grad_(False)
                gaussians.setup_rendering_learning_rate()
            gaussians.update_learning_rate(iteration - warm_iter)
        else:
            gaussians.update_warm_start_learning_rate(iteration)

        if args.optim_pose==False:
            gaussians.P.requires_grad_(False)

        # (DISABLED) Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        pose = gaussians.get_RT(viewpoint_cam.uid)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        gaussians.inference()

        pretrained_loss_dict = {
            'xyz': l1_loss(gaussians._xyz, gaussians.param_init['xyz']),
            'f_dc': l1_loss(gaussians._features_dc, gaussians.param_init['f_dc']),
            'f_rest': l1_loss(gaussians._features_rest, gaussians.param_init['f_rest']),
            'opacity': l1_loss(gaussians._opacity, gaussians.param_init['opacity']),
            'scaling': l1_loss(gaussians._scaling, gaussians.param_init['scaling']),
            'rotation': l1_loss(gaussians._rotation, gaussians.param_init['rotation']),
            'pose': l1_loss(gaussians.P, gaussians.param_init['pose']),
            # 'focal': l1_loss(gaussians._focal_params, gaussians.param_init['focal']),
            'pc_feat':l1_loss(gaussians.pc_feat, gaussians.param_init['pc_feat']),
            }

        if iteration <= warm_iter:
            loss = sum(loss for key, loss in pretrained_loss_dict.items() if key in gs_params_group['head'])
            Ll1 = torch.tensor(0)

        if iteration > warm_iter:
            render_pkg = render_gsplat(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 

            if feat_type in ['iuv', 'iuvrgb']:
                # Add scaling regularization for 'iuv' and 'iuvrgb' features
                # Prevents their gaussians scale from becoming too large to cause CUDA out of memory
                loss += l1_loss(gaussians._scaling, gaussians.param_init['scaling']) * 0.1

        loss.backward()
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
            training_report(
                tb_writer, iteration, Ll1, loss, l1_loss, 
                iter_start.elapsed_time(iter_end), testing_iterations, 
                scene, render_gsplat, (pipe, background), 
                pretrained_loss_dict, 
                save_metrics_iter=[opt.iterations]
                )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(scene.model_path + 'pose' + f"/pose_{iteration}.npy", gaussians.P, train_cams_init)

            # (DISABLED) Densification
            # if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
        end = perf_counter()
        train_time = end - start

    # We commented out log&save operations, and then calculate train time.
    # train_time = np.array(train_time)
    # print("total_test_time_epoch: ", 1)
    # print("instantsplat_train_time_mean: ", train_time.mean())
    # print("instantsplat_train_time_median: ", np.median(train_time))

def prepare_output_and_logger(args, iteration=None):    
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
        tb_writer = SummaryWriter(os.path.join(args.model_path, f"log_{iteration}"))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, prop_pred=None, save_metrics_iter=None):
    if tb_writer:
        tb_writer.add_scalar('train_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for key, values in prop_pred.items():
            tb_writer.add_scalar(f'train_patches/delta_{key}', values.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name']=="train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, camera_pose=pose)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                    with torch.no_grad():
                        pred_xyz = scene.gaussians._xyz.detach()
                        pred_opacity = scene.gaussians.get_opacity.detach()
                        gt_xyz = scene.gaussians.gt_xyz
                        
                        # thresholds = [0.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5] if iteration in save_metrics_iter else [0.0]
                        # metrics_dict = compute_3d_metrics(pred_xyz, gt_xyz, pred_opacity, thresholds=thresholds)
                        metrics_dict = compute_3d_metrics(pred_xyz, gt_xyz, pred_opacity)

                        metrics_t0 = metrics_dict[0.0]
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - Accuracy', metrics_t0['Accuracy'], iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - Acc_med', metrics_t0['Acc_med'], iteration) 
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - Completion', metrics_t0['Completion'], iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - Comp_med', metrics_t0['Comp_med'], iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - Distance', metrics_t0['Distance'], iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - Dist_med', metrics_t0['Dist_med'], iteration)

                        if iteration in save_metrics_iter:
                            results_dict = {}
                            metrics_path = os.path.join(scene.model_path, "3d_metrics.json")
                            if os.path.exists(metrics_path):
                                with open(metrics_path, 'r') as fp:
                                    results_dict = json.load(fp)
                            
                            results_dict[f"ours_{iteration}"] = {
                                f"threshold_{threshold}": {
                                    k: float(v) for k, v in metrics.items()
                                }
                                for threshold, metrics in metrics_dict.items()
                            }
                            
                            with open(metrics_path, 'w') as fp:
                                json.dump(results_dict, fp, indent=True)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
    parser.add_argument("--test_iterations", nargs="+", type=int, 
                        default=[500, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7_000, \
                                 8_000, 9_000, 10_000, 11_000, 12_000, 13_000, 14_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--n_views", type=int, default=None)
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--optim_pose", action="store_true")
    parser.add_argument("--feat_type", type=str, nargs='*', default=None, help="Feature type(s). Multiple types can be specified for combination.")
    parser.add_argument("--method", type=str, default='dust3r', help="Method of Initialization, e.g., 'dust3r' or 'mast3r'")
    parser.add_argument("--feat_dim", type=int, default=None, help="Feture dimension after PCA . If None, PCA is not applied.")
    parser.add_argument("--model", type=str, default='G', help="Model of Feat2gs, 'G'='geometry'/'T'='texture'/'A'='all'")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Noise for gt pointcloud positions")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")

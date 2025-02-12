import os
import shutil
import torch
import numpy as np
import argparse
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r")))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r", "dust3r")))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from utils.dust3r_utils import  (load_images, save_colmap_images, rigid_points_registration)
from utils.feat_utils import InitMethod
from utils.loss_utils import calculate_in_frustum_mask
from dust3r.utils.device import todevice

import torch.nn.functional as F
import torchvision


def get_args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="Path to the model checkpoint")
    parser.add_argument("--model_path", type=str, default="submodules/mast3r/checkpoints/", help="Directory with models")
    parser.add_argument("--device", type=str, default='cuda', help="Device for inference")
    parser.add_argument("--focal_avg", action="store_true", help="Use averaging focal")
    parser.add_argument("--llffhold", type=int, default=2, help="Hold out every n-th image from the dataset.")
    parser.add_argument("--n_views", type=int, default=3, help="Number of views to use")
    parser.add_argument("--img_base_path", type=str, required=True, help="Directory with images")
    parser.add_argument('--min_conf_thr', type=float, default=0.0, help="Minimum confidence threshold")
    parser.add_argument('--tsdf_thresh', type=float, default=0.0, help="TSDF threshold")
    parser.add_argument("--method", type=str, default='dust3r', help="Method of Initialization, e.g., 'dust3r' or 'mast3r'")
    parser.add_argument("--feat_type", type=str, nargs='*', default=None, help="Feature type(s). Multiple types can be specified for combination.")

    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    device = args.device
    n_views = args.n_views
    method = args.method
    feat_type = args.feat_type
    img_base_path = args.img_base_path

    all_img_folder = os.path.join(img_base_path, "images")
    train_img_folder = os.path.join(img_base_path, f"{n_views}_views/images")

    test_view_folder=train_img_folder.replace("images", f"test_view/sparse/0/{method}")
    test_pose_path = os.path.join(test_view_folder, 'images.txt')
    mask_output_path = train_img_folder.replace("images", f"test_view/masks")

    # ---------------- (1) Prepare Train & Test images list ---------------- 
    all_img_list = sorted(os.listdir(all_img_folder))
    train_img_list = sorted(os.listdir(train_img_folder))
    assert len(train_img_list)==n_views, f"Number of images in the folder is not equal to {n_views}"
    test_img_list = [img for img in all_img_list if img not in train_img_list]
    all_img_list = train_img_list + test_img_list
    all_img_list = [os.path.join(all_img_folder, img) for img in all_img_list]
    
    # Check if the test_pose_path file is not empty
    test_pose_exists_and_not_empty = os.path.exists(test_pose_path) and os.path.getsize(test_pose_path) > 0

    if not os.path.exists(mask_output_path):
        all_mask_exist = False
    else:
        # Check if the target mask path exists
        all_mask_exist = all(os.path.exists(os.path.join(mask_output_path, img_name)) for img_name in test_img_list)
        all_mask_exist *= all(img_name in test_img_list for img_name in os.listdir(mask_output_path))

    # If all target paths exist and the test_pose_path is not empty, skip the remaining code
    if all_mask_exist and test_pose_exists_and_not_empty:
        print("All target files and the test poses already exist. Exiting...")
        exit()
    else:
        # if os.path.exists(test_view_folder):
        #     shutil.rmtree(test_view_folder)
        if os.path.exists(mask_output_path):
            shutil.rmtree(mask_output_path)
        os.makedirs(test_view_folder, exist_ok=True)
        os.makedirs(mask_output_path, exist_ok=True)

    #---------------- (2) Load train pointcloud and intrinsic (define as m1) ---------------- 
    train_pts_all_path = os.path.join(img_base_path, f"{n_views}_views/sparse/0/{method}", "pts_4_3dgs_all.npy")
    if feat_type:
        train_pts_all_path = train_pts_all_path.replace("pts_4_3dgs_all.npy", f"{'-'.join(feat_type)}/pts_4_3dgs_all.npy")
    train_pts_all = np.load(train_pts_all_path)
    train_pts3d_m1 = train_pts_all
    
    if args.focal_avg:
        focal_path = train_pts_all_path.replace("pts_4_3dgs_all.npy", "focal.npy")
        preset_focal = np.load(focal_path) # load focal calculated by dust3r_coarse_geometry_initialization

    #---------------- (3) Get N_views pointcloud and test pose (define as n1) ----------------     
    # load model
    args.img_base_path = os.path.join(img_base_path, f"{n_views}_views/test_view")
    init_method = InitMethod(args)
    model = init_method.get_model()

    # read all images
    images, ori_size = load_images(all_img_list, size=512)
    print("ori_size", ori_size)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)    
    scene = init_method.infer(pairs=pairs, model=model, train_img_list=sorted(os.listdir(all_img_folder)), 
                              known_focal=preset_focal[0][0])
    _, _, all_poses, all_intrinsics, all_pts3d, _= init_method.get_info(scene)
    all_depth = torch.stack(todevice(init_method.get_depth(scene), device))[None]
  
    train_pts3d_n1 = all_pts3d[:n_views] 
    test_poses_n1 = all_poses[n_views:] 

    train_pts3d_n1 = np.array(to_numpy(train_pts3d_n1)).reshape(-1,3)
    test_poses_n1  = np.array(to_numpy(test_poses_n1))  # test_pose_n1: c2w

    #---------------- (4) Applying pointcloud registration & Calculate transform_matrix and visible masks & Save initial_test_pose and masks---------------- 
    # compute transform that goes from cam to world
    train_pts3d_n1 = torch.from_numpy(train_pts3d_n1)
    train_pts3d_m1 = torch.from_numpy(train_pts3d_m1)
    s, R, T = rigid_points_registration(train_pts3d_n1, train_pts3d_m1)

    transform_matrix = torch.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T
    transform_matrix[:3, 3] *= s
    transform_matrix = transform_matrix.numpy()

    test_poses_m1 = transform_matrix @ test_poses_n1
    save_colmap_images(test_poses_m1, test_pose_path, test_img_list)

    # Compute visibility masks following Splatt3R
    all_intrinsics = todevice(all_intrinsics, device)[None]
    all_poses = todevice(all_poses, device)[None]
    masks = calculate_in_frustum_mask(
        all_depth[:, n_views:], all_intrinsics[:,n_views:], all_poses[:,n_views:],
        all_depth[:,:n_views], all_intrinsics[:,:n_views], all_poses[:,:n_views]
    )

    upsampled_masks = F.interpolate(masks.float(), size=(ori_size[1], ori_size[0]), mode='nearest')[0]

    for idx, (mask, name) in enumerate(tqdm(zip(upsampled_masks, test_img_list), desc="Saving mask")):
        torchvision.utils.save_image(
            mask, os.path.join(mask_output_path, name)
        )


  
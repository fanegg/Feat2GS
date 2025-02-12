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

from utils.dust3r_utils import load_images_dtu, save_colmap_images
from utils.loss_utils import calculate_in_frustum_mask
from dust3r.utils.device import todevice

import torch.nn.functional as F
import torchvision


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda', help="Device for inference")
    parser.add_argument("--n_views", type=int, default=3, help="Number of views to use")
    parser.add_argument("--img_base_path", type=str, required=True, help="Directory with images")
    parser.add_argument("--method", type=str, default='gt', help="Method of Initialization")

    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    device = args.device
    n_views = args.n_views
    method = args.method
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


    #---------------- (3) Get N_views pointcloud and test pose ----------------     
    # read all images
    images, ori_size = load_images_dtu(all_img_list, size=512, scene_folder=img_base_path)
    print("ori_size", ori_size)

    all_intrinsics = np.stack([img['camera_intrinsics'] for img in images])
    all_poses = np.stack([img['camera_pose'] for img in images])
    all_depth = np.stack([img['depthmap'] for img in images])
 
    #TODO: remove this
    all_depth = all_depth / 1000.0
    all_poses[..., :3, 3] = all_poses[..., :3, 3] / 1000.0

    test_poses = all_poses[n_views:] 

    #---------------- (4) Save initial_test_pose---------------- 
    save_colmap_images(test_poses, test_pose_path, test_img_list)

    all_intrinsics = todevice(all_intrinsics, device)[None]
    all_poses = todevice(all_poses, device)[None]
    all_depth = todevice(all_depth, device)[None]

    masks = calculate_in_frustum_mask(
        all_depth[:, n_views:], all_intrinsics[:,n_views:], all_poses[:,n_views:],
        all_depth[:,:n_views], all_intrinsics[:,:n_views], all_poses[:,:n_views],
        atol=0.01
    )

    #TODO: use this
    #atol=10.

    upsampled_masks = F.interpolate(masks.float(), size=(ori_size[1], ori_size[0]), mode='nearest')[0]

    for idx, (mask, name) in enumerate(tqdm(zip(upsampled_masks, test_img_list), desc="Saving mask")):
        torchvision.utils.save_image(
            mask, os.path.join(mask_output_path, name)
        )


  
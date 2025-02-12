import os
import numpy as np
import argparse
import time
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r")))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r", "dust3r")))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from dust3r.image_pairs import make_pairs
from utils.dust3r_utils import load_images, load_images_dtu, storePly, save_colmap_cameras, save_colmap_images
from utils.feat_utils import FeatureExtractor, InitMethod

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
    parser.add_argument("--feat_dim", type=int, default=None, help="PCA dimension. If None, PCA is not applied, and the original feature dimension is retained.")
    parser.add_argument("--feat_type", type=str, nargs='*', default=None, help="Feature type(s). Multiple types can be specified for combination.")
    # parser.add_argument("--use_featup", action="store_true", help="Use FeatUp for upsampling")
    parser.add_argument("--vis_feat", action="store_true", help="Visualize features")
    parser.add_argument("--vis_key", type=str, default=None, help="Feature type to visualize, e.g., 'decfeat' or 'desc'")
    parser.add_argument("--method", type=str, default='gt', help="Method of Initialization")

    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    device = args.device
    n_views = args.n_views
    method = args.method
    img_base_path = args.img_base_path
    img_folder_path = os.path.join(img_base_path, f"{n_views}_views/images")
    args.img_base_path = os.path.join(img_base_path, f"{n_views}_views")
    
    # # Load model and images
    # init_method = InitMethod(args)
    # model = init_method.get_model()

    train_img_list = sorted(os.listdir(img_folder_path))
    assert len(train_img_list)==n_views, f"Number of images in the folder is not equal to {n_views}"

    images, ori_size = load_images_dtu(img_folder_path, size=512, scene_folder=img_base_path)
    print("ori_size", ori_size)

    start_time = time.time()
    ##############################################################################################################################################################
    # Generate pairs and run inference
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    # scene = init_method.infer(pairs=pairs, model=model, train_img_list=train_img_list)
    # # Extract scene information
    # imgs, focals, poses, intrinsics, pts3d, confidence_masks= init_method.get_info(scene)

    # For DTU dataset, we use pointclouds GT and noised camera param GT
    intrinsics = np.stack([img['camera_intrinsics'] for img in images])
    poses = np.stack([img['camera_pose'] for img in images])
    pts3d = [img['pts3d'] for img in images]

    # For DTU dataset, scale down points and poses by 1000.0
    pts3d = [img['pts3d'] / 1000.0 for img in images]
    poses[..., :3, 3] = poses[..., :3, 3] / 1000.0

    confidence_masks = [img['valid_mask'] for img in images]
    imgs = [(img['img'][0].cpu().numpy().transpose(1, 2, 0) + 1) / 2 for img in images]

    torch.cuda.empty_cache()
    
    ##############################################################################################################################################################
    end_time = time.time()
    print(f"Time taken for {n_views} views: {end_time-start_time} seconds")

    output_colmap_path=img_folder_path.replace("images", f"sparse/0/{method}")

    if args.feat_type:
        extractor = FeatureExtractor(images, args, method)
        feats = extractor(pairs=pairs, train_img_list=train_img_list)

        feat_type_str = '-'.join(extractor.feat_type)
        output_colmap_path = os.path.join(output_colmap_path, feat_type_str)

    os.makedirs(output_colmap_path, exist_ok=True)
    print(f"The initial reconstruction will be saved at >>>>> {output_colmap_path}")

    # Save
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)

    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)

    if args.feat_type:
        feat_4_3dgs = np.concatenate([p[m] for p, m in zip(feats, confidence_masks)])
        storePly(os.path.join(output_colmap_path, f"points3D.ply"), pts_4_3dgs, color_4_3dgs, feat_4_3dgs)
    else:
        storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)

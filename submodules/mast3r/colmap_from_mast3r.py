import matplotlib
matplotlib.use('Agg')

import sys
import argparse
import torch
import numpy as np
import os
import re
import cv2
import trimesh
from pathlib import Path
from PIL import Image
from typing import NamedTuple, Optional

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'mast3r'))
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
import mast3r.utils.path_to_dust3r

from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs

from plyfile import PlyData, PlyElement

from utils.dust3r_utils import visualizer, pca, upsampler

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def invert_matrix(mat):
    """Invert a torch or numpy matrix."""
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'Unsupported matrix type: {type(mat)}')

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# Ensure save directories exist
def init_filestructure(save_path):
    save_path.mkdir(exist_ok=True, parents=True)
    images_path = save_path / 'images'
    masks_path = save_path / 'masks'
    sparse_path = save_path / 'sparse/0'
    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)
    sparse_path.mkdir(exist_ok=True, parents=True)
    return save_path, images_path, masks_path, sparse_path

# Save images and masks
def save_images_and_masks(imgs, masks, images_path, img_files, masks_path):
    for i, (image, name, mask) in enumerate(zip(imgs, img_files, masks)):
        imgname = Path(name).stem
        image_save_path = images_path / f"{imgname}.png"
        mask_save_path = masks_path / f"{imgname}.png"
        rgb_image = cv2.cvtColor(image * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_save_path), rgb_image)
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=2) * 255
        Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)

# Save camera information
def save_cameras(focals, principal_points, sparse_path, imgs_shape):
    cameras_file = sparse_path / 'cameras.txt'
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (focal, pp) in enumerate(zip(focals, principal_points)):
            f.write(f"{i} PINHOLE {imgs_shape[2]} {imgs_shape[1]} {focal} {focal} {pp[0]} {pp[1]}\n")

# Save image transformations
def save_images_txt(world2cam, img_files, sparse_path):
    images_file = sparse_path / 'images.txt'
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(world2cam.shape[0]):
            name = Path(img_files[i]).stem
            rotation_matrix = world2cam[i, :3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = world2cam[i, :3, 3]
            f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {name}.png\n\n")

# Save point cloud with normals
def save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path):
    pc = get_point_cloud(imgs, pts3d, masks)
    default_normal = [0, 1, 0]
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))
    save_path = sparse_path / 'points3D.ply'
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))
    with open(save_path, 'w') as f:
        f.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {int(color[0])} {int(color[1])} {int(color[2])} {normal[0]} {normal[1]} {normal[2]}\n")

# Generate point cloud
def get_point_cloud(imgs, pts3d, mask):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask.reshape(mask.shape[0], -1))])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    pts = pts.reshape(-1, 3)[::3]
    col = col.reshape(-1, 3)[::3]
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals
    return pct

def main(image_dir, save_dir, model_path, device, batch_size, image_size, schedule, lr, niter, min_conf_thr, tsdf_thresh):
    # Load model and images
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    image_files = sorted([str(x) for x in Path(image_dir).iterdir() if x.suffix in ['.png', '.jpg']],
                         key=lambda x: int(re.search(r'\d+', Path(x).stem).group()))
    images = load_images(image_files, size=image_size)

    # Generate pairs and run inference
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

    cache_dir = os.path.join(save_dir, 'cache')
    if os.path.exists(cache_dir):
        os.system(f'rm -rf {cache_dir}')
    scene = sparse_global_alignment(image_files, pairs, cache_dir,
                                    model, lr1=0.07, niter1=500, lr2=0.014, niter2=200, device=device,
                                    opt_depth='depth' in 'refine', shared_intrinsics=False,
                                    matching_conf_thr=5.)

    # Extract scene information
    world2cam = invert_matrix(scene.get_im_poses().detach()).cpu().numpy()
    principal_points = scene.get_principal_points().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    imgs = np.array(scene.imgs)

    tsdf = TSDFPostProcess(scene, TSDF_thresh=tsdf_thresh)
    pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=True))
    masks = np.array(to_numpy([c > min_conf_thr for c in confs]))

    _, H, W, _ = imgs.shape
    feat_dim = 64
    projected_feat = pca(scene.stacked_feat, feat_dim)
    upsampled_feat = upsampler(projected_feat, H, W)
    visualizer(upsampled_feat, images, save_dir)

    # Main execution
    save_path, images_path, masks_path, sparse_path = init_filestructure(Path(save_dir))
    save_images_and_masks(imgs, masks, images_path, image_files, masks_path)
    save_cameras(focals, principal_points, sparse_path, imgs_shape=imgs.shape)
    save_images_txt(world2cam, image_files, sparse_path)
    save_pointcloud_with_normals(imgs, pts3d, masks, save_dir)

    print(f'[INFO] Mast3R Reconstruction is successfully converted to COLMAP files in: {str(sparse_path)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing images')
    parser.add_argument('--image_size', type=int, default=512, help='Size to resize images')
    parser.add_argument('--schedule', type=str, default='cosine', help='Learning rate schedule')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--niter', type=int, default=300, help='Number of iterations')
    parser.add_argument('--min_conf_thr', type=float, default=1.5, help='Minimum confidence threshold')
    parser.add_argument('--tsdf_thresh', type=float, default=0.0, help='TSDF threshold')

    args = parser.parse_args()
    main(args.image_dir, args.save_dir, args.model_path, args.device, args.batch_size, args.image_size, args.schedule, args.lr, args.niter, args.min_conf_thr, args.tsdf_thresh)
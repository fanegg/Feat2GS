import os
import numpy as np
import sys
import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MAST3R_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "submodules", "mast3r"))
DUST3R_DIR = os.path.abspath(os.path.join(MAST3R_DIR, "dust3r"))

sys.path.extend([
    PROJECT_DIR,
    MAST3R_DIR,
    DUST3R_DIR
])

from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from dust3r.utils.image import imread_cv2


def load_cam_mvsnet(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = 192
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0
    
    
    extrinsic = cam[0].astype(np.float32)
    intrinsic = cam[1].astype(np.float32)

    return intrinsic, extrinsic


def process_dtu(image_path, dtu_base):
    """Process a single DTU dataset image with its corresponding depth map and camera parameters
    
    Args:
        image_path (str): Full path to the image file
    """
    root = os.path.dirname(image_path)
    base_dir = os.path.dirname(os.path.dirname(root))
    filename = os.path.basename(image_path)
    
    depth_path = os.path.join(dtu_base, 'depths', filename.replace('.jpg', '.npy'))
    cam_path = os.path.join(dtu_base, 'cams', filename.replace('.jpg', '_cam.txt'))
    mask_path = os.path.join(dtu_base, 'binary_masks', filename.replace('.jpg', '.png'))

    rgb_image = imread_cv2(str(image_path))
    H1, W1 = rgb_image.shape[:2]

    depthmap = np.load(depth_path)
    depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)

    mask = imread_cv2(mask_path, cv2.IMREAD_UNCHANGED)/255.0
    mask = mask.astype(np.float32)

    mask[mask>0.5] = 1.0
    mask[mask<0.5] = 0.0

    mask = cv2.resize(mask, (depthmap.shape[1], depthmap.shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((10, 10), np.uint8)  # Define the erosion kernel
    mask = cv2.erode(mask, kernel, iterations=1)
    depthmap = depthmap * mask
    
    cur_intrinsics, camera_pose = load_cam_mvsnet(open(cam_path, 'r'))
    intrinsics = cur_intrinsics[:3, :3]
    camera_pose = np.linalg.inv(camera_pose)
    
    img = dict(
        depthmap=depthmap,
        camera_pose=camera_pose,
        camera_intrinsics=intrinsics
    ) 

    pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**img)
    valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)
    masked_rgb = rgb_image * valid_mask[..., None]
    
    return masked_rgb

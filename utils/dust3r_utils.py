import os
import torch
import cv2
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
from plyfile import PlyData, PlyElement
import torchvision.transforms as tvf
import roma
import dust3r.cloud_opt.init_im_poses as init_fun
from dust3r.cloud_opt.base_opt import global_alignment_loop
from dust3r.utils.geometry import geotrf, inv, depthmap_to_absolute_camera_coordinates
from dust3r.cloud_opt.commons import edge_str
from dust3r.utils.image import _resize_pil_image, imread_cv2
import dust3r.datasets.utils.cropping as cropping
import torch.nn.functional as F

def get_known_poses(scene):
        if scene.has_im_poses:
            known_poses_msk = torch.tensor([not (p.requires_grad) for p in scene.im_poses])
            known_poses = scene.get_im_poses()
            return known_poses_msk.sum(), known_poses_msk, known_poses
        else:
            return 0, None, None

def init_from_pts3d(scene, pts3d, im_focals, im_poses):
    # init poses
    nkp, known_poses_msk, known_poses = get_known_poses(scene)
    if nkp == 1:
        raise NotImplementedError("Would be simpler to just align everything afterwards on the single known pose")
    elif nkp > 1:
        # global rigid SE3 alignment
        s, R, T = init_fun.align_multiple_poses(im_poses[known_poses_msk], known_poses[known_poses_msk])
        trf = init_fun.sRT_to_4x4(s, R, T, device=known_poses.device)

        # rotate everything
        im_poses = trf @ im_poses
        im_poses[:, :3, :3] /= s  # undo scaling on the rotation part
        for img_pts3d in pts3d:
            img_pts3d[:] = geotrf(trf, img_pts3d)

    # set all pairwise poses
    for e, (i, j) in enumerate(scene.edges):
        i_j = edge_str(i, j)
        # compute transform that goes from cam to world
        s, R, T = init_fun.rigid_points_registration(scene.pred_i[i_j], pts3d[i], conf=scene.conf_i[i_j])
        scene._set_pose(scene.pw_poses, e, R, T, scale=s)

    # take into account the scale normalization
    s_factor = scene.get_pw_norm_scale_factor()
    im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
    for img_pts3d in pts3d:
        img_pts3d *= s_factor

    # init all image poses
    if scene.has_im_poses:
        for i in range(scene.n_imgs):
            cam2world = im_poses[i]
            depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
            scene._set_depthmap(i, depth)
            scene._set_pose(scene.im_poses, i, cam2world)
            if im_focals[i] is not None:
                scene._set_focal(i, im_focals[i])

    if scene.verbose:
        print(' init loss =', float(scene()))

@torch.no_grad()
def init_minimum_spanning_tree(scene, focal_avg=False, known_focal=None, **kw):
    """ Init all camera poses (image-wise and pairwise poses) given
        an initial set of pairwise estimations.
    """
    device = scene.device
    pts3d, _, im_focals, im_poses = init_fun.minimum_spanning_tree(scene.imshapes, scene.edges,
                                                        scene.pred_i, scene.pred_j, scene.conf_i, scene.conf_j, scene.im_conf, scene.min_conf_thr,
                                                        device, has_im_poses=scene.has_im_poses, verbose=scene.verbose,
                                                        **kw)

    if known_focal is not None:
        repeat_focal = np.repeat(known_focal, len(im_focals))
        for i in range(len(im_focals)):
            im_focals[i] = known_focal
        scene.preset_focal(known_focals=repeat_focal)
    elif focal_avg:
        im_focals_avg = np.array(im_focals).mean()
        for i in range(len(im_focals)):
            im_focals[i] = im_focals_avg
        repeat_focal = np.array(im_focals)#.cpu().numpy()
        scene.preset_focal(known_focals=repeat_focal)

    return init_from_pts3d(scene, pts3d, im_focals, im_poses)

@torch.cuda.amp.autocast(enabled=False)
def compute_global_alignment(scene, init=None, niter_PnP=10, focal_avg=False, known_focal=None, **kw):
    if init is None:
        pass
    elif init == 'msp' or init == 'mst':
        init_minimum_spanning_tree(scene, niter_PnP=niter_PnP, focal_avg=focal_avg, known_focal=known_focal)
    elif init == 'known_poses':
        init_fun.init_from_known_poses(scene, min_conf_thr=scene.min_conf_thr,
                                        niter_PnP=niter_PnP)
    else:
        raise ValueError(f'bad value for {init=}')

    return global_alignment_loop(scene, **kw)



def load_images(folder_or_list, size, square_ok=False):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    imgs = []
    for path in folder_content:
        if not path.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        W2 = W//16*16
        H2 = H//16*16
        img = np.array(img)
        img = cv2.resize(img, (W2,H2), interpolation=cv2.INTER_LINEAR)
        img = PIL.Image.fromarray(img)

        print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    print(f' (Found {len(imgs)} images)')
    return imgs, (W1,H1)


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


def _crop_resize_if_necessary(image, depthmap, intrinsics, resolution, rng=None, info=None):
    """ This function:
        - first downsizes the image with LANCZOS inteprolation,
            which is better than bilinear interpolation in
    """
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)

    # downscale with lanczos interpolation so that image.size == resolution
    # cropping centered on the principal point
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    
    # calculate min distance to margin
    min_margin_x = min(cx, W-cx)
    min_margin_y = min(cy, H-cy)
    assert min_margin_x > W/5, f'Bad principal point in view={info}'
    assert min_margin_y > H/5, f'Bad principal point in view={info}'
    
    ## Center crop
    # Crop on the principal point, make it always centered
    # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    # transpose the resolution if necessary
    W, H = image.size  # new size
    assert resolution[0] >= resolution[1]
    if H > 1.1*W:
        # image is portrait mode
        resolution = resolution[::-1]
    elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
        # image is square, so we chose (portrait, landscape) randomly
        if rng.integers(2):
            resolution = resolution[::-1]

    # high-quality Lanczos down-scaling
    target_resolution = np.array(resolution)
    
    ## Recale with max factor, so  one of width or height might be larger than target_resolution
    image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

    # actual cropping (if necessary) with bilinear interpolation
    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    return image, depthmap, intrinsics2


def load_images_dtu(folder_or_list, size, scene_folder):
    """ 
    Preprocessing DTU requires depth, camera param and mask.
    We follow Splatt3R to compute valid_mask.
    """
    if isinstance(folder_or_list, str):
        print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        print(f'>> Loading a list of {len(folder_or_list)} images')
        root = os.path.dirname(folder_or_list[0]) if folder_or_list else ''
        folder_content = [os.path.basename(p) for p in folder_or_list]

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    depth_root = os.path.join(scene_folder, 'depths')
    mask_root = os.path.join(scene_folder, 'binary_masks')
    cam_root = os.path.join(scene_folder, 'cams')

    imgs = []
    for path in folder_content:
        if not path.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            continue

        impath = os.path.join(root, path)
        depthpath = os.path.join(depth_root, path.replace('.jpg', '.npy'))
        campath = os.path.join(cam_root, path.replace('.jpg', '_cam.txt'))
        maskpath = os.path.join(mask_root, path.replace('.jpg', '.png'))

        rgb_image = imread_cv2(impath)
        H1, W1 = rgb_image.shape[:2]
        depthmap = np.load(depthpath)
        depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)

        mask = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED)/255.0
        mask = mask.astype(np.float32)

        mask[mask>0.5] = 1.0
        mask[mask<0.5] = 0.0

        mask = cv2.resize(mask, (depthmap.shape[1], depthmap.shape[0]), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((10, 10), np.uint8)  # Define the erosion kernel
        mask = cv2.erode(mask, kernel, iterations=1)
        depthmap = depthmap * mask
        
        cur_intrinsics, camera_pose = load_cam_mvsnet(open(campath, 'r'))
        intrinsics = cur_intrinsics[:3, :3]
        camera_pose = np.linalg.inv(camera_pose)

        new_size = tuple(int(round(x*size/max(W1, H1))) for x in (W1, H1))
        W, H = new_size
        W2 = W//16*16
        H2 = H//16*16
        
        rgb_image, depthmap, intrinsics = _crop_resize_if_necessary(
            rgb_image, depthmap, intrinsics, (W2, H2), info=impath)
        
        print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        img = dict(
            img=ImgNorm(rgb_image)[None], 
            true_shape=np.int32([rgb_image.size[::-1]]), 
            idx=len(imgs), 
            instance=str(len(imgs)),
            depthmap=depthmap,
            camera_pose=camera_pose,
            camera_intrinsics=intrinsics
        ) 

        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**img)
        img['pts3d'] = pts3d
        img['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

        imgs.append(img)


    assert imgs, 'no images foud at '+root
    print(f' (Found {len(imgs)} images)')
    return imgs, (W1,H1)


def storePly(path, xyz, rgb, feat=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    if feat is not None:
        for i in range(feat.shape[1]):
            dtype.append((f'feat_{i}', 'f4'))

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)

    if feat is not None:
        attributes = np.concatenate((attributes, feat), axis=1)

    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def R_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.

    Parameters:
    - R: A 3x3 numpy array representing a rotation matrix.

    Returns:
    - A numpy array representing the quaternion [w, x, y, z].
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return np.array([w, x, y, z])

def save_colmap_cameras(ori_size, intrinsics, camera_file):
    with open(camera_file, 'w') as f:
        for i, K in enumerate(intrinsics, 1):  # Starting index at 1
            width, height = ori_size
            scale_factor_x = width/2  / K[0, 2]
            scale_factor_y = height/2  / K[1, 2]
            # assert scale_factor_x==scale_factor_y, "scale factor is not same for x and y"
            # print(f'scale factor is not same for x {scale_factor_x} and y {scale_factor_y}')
            f.write(f"{i} PINHOLE {width} {height} {K[0, 0]*scale_factor_x} {K[1, 1]*scale_factor_x} {width/2} {height/2}\n") # scale focal
            # f.write(f"{i} PINHOLE {width} {height} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")

def save_colmap_images(poses, images_file, train_img_list):
    with open(images_file, 'w') as f:
        for i, pose in enumerate(poses, 1):  # Starting index at 1
            # breakpoint()
            pose = np.linalg.inv(pose)
            R = pose[:3, :3]
            t = pose[:3, 3]
            q = R_to_quaternion(R)  # Convert rotation matrix to quaternion
            f.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i} {os.path.basename(train_img_list[i-1])}\n")
            f.write(f"\n")


def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded


def rigid_points_registration(pts1, pts2, conf=None):
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf, compute_scaling=True)
    return s, R, T  # return un-scaled (R, T)

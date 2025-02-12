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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import einops

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def masked_ssim(img1, img2, mask):
    ssim_map = ssim(img1, img2, get_ssim_map=True)
    return (ssim_map * mask).sum() / (3. * mask.sum())


def ssim(img1, img2, window_size=11, size_average=True, get_ssim_map=False):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, get_ssim_map)

def _ssim(img1, img2, window, window_size, channel, size_average=True, get_ssim_map=False):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if get_ssim_map:
        return ssim_map
    elif size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# --- Projections ---

def homogenize_points(points):
    """Append a '1' along the final dimension of the tensor (i.e. convert xyz->xyz1)"""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def normalize_homogenous_points(points):
    """Normalize the point vectors"""
    return points / points[..., -1:]


def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    """
    Convert pixel space points to camera space points.

    Args:
        pixel_space_points (torch.Tensor): Pixel space points with shape (h, w, 2)
        depth (torch.Tensor): Depth map with shape (b, v, h, w, 1)
        intrinsics (torch.Tensor): Camera intrinsics with shape (b, v, 3, 3)

    Returns:
        torch.Tensor: Camera space points with shape (b, v, h, w, 3).
    """
    pixel_space_points = homogenize_points(pixel_space_points)
    camera_space_points = torch.einsum('b v i j , h w j -> b v h w i', intrinsics.inverse(), pixel_space_points)
    camera_space_points = camera_space_points * depth
    return camera_space_points


def camera_space_to_world_space(camera_space_points, c2w):
    """
    Convert camera space points to world space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum('b v i j , b v h w j -> b v h w i', c2w, camera_space_points)
    return world_space_points[..., :3]


def camera_space_to_pixel_space(camera_space_points, intrinsics):
    """
    Convert camera space points to pixel space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v1, v2, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 3, 3)

    Returns:
        torch.Tensor: World space points with shape (b, v1, v2, h, w, 2).
    """
    camera_space_points = normalize_homogenous_points(camera_space_points)
    pixel_space_points = torch.einsum('b u i j , b v u h w j -> b v u h w i', intrinsics, camera_space_points)
    return pixel_space_points[..., :2]


def world_space_to_camera_space(world_space_points, c2w):
    """
    Convert world space points to pixel space points.

    Args:
        world_space_points (torch.Tensor): World space points with shape (b, v1, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, v2, h, w, 3).
    """
    world_space_points = homogenize_points(world_space_points)
    camera_space_points = torch.einsum('b u i j , b v h w j -> b v u h w i', c2w.inverse(), world_space_points)
    return camera_space_points[..., :3]


def unproject_depth(depth, intrinsics, c2w):
    """
    Turn the depth map into a 3D point cloud in world space

    Args:
        depth: (b, v, h, w, 1)
        intrinsics: (b, v, 3, 3)
        c2w: (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """

    # Compute indices of pixels
    h, w = depth.shape[-3], depth.shape[-2]
    x_grid, y_grid = torch.meshgrid(
        torch.arange(w, device=depth.device, dtype=torch.float32),
        torch.arange(h, device=depth.device, dtype=torch.float32),
        indexing='xy'
    )  # (h, w), (h, w)

    # Compute coordinates of pixels in camera space
    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)  # (..., h, w, 2)
    camera_points = pixel_space_to_camera_space(pixel_space_points, depth, intrinsics)  # (..., h, w, 3)

    # Convert points to world space
    world_points = camera_space_to_world_space(camera_points, c2w)  # (..., h, w, 3)

    return world_points


@torch.no_grad()
def calculate_in_frustum_mask(depth_1, intrinsics_1, c2w_1, depth_2, intrinsics_2, c2w_2, atol=1e-2):
    """
    A function that takes in the depth, intrinsics and c2w matrices of two sets
    of views, and then works out which of the pixels in the first set of views
    has a direct corresponding pixel in any of views in the second set

    Args:
        depth_1: (b, v1, h, w)
        intrinsics_1: (b, v1, 3, 3)
        c2w_1: (b, v1, 4, 4)
        depth_2: (b, v2, h, w)
        intrinsics_2: (b, v2, 3, 3)
        c2w_2: (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, h, w).
    """

    _, v1, h, w = depth_1.shape
    _, v2, _, _ = depth_2.shape

    # Unproject the depth to get the 3D points in world space
    points_3d = unproject_depth(depth_1[..., None], intrinsics_1, c2w_1)  # (b, v1, h, w, 3)

    # Project the 3D points into the pixel space of all the second views simultaneously
    camera_points = world_space_to_camera_space(points_3d, c2w_2)  # (b, v1, v2, h, w, 3)
    points_2d = camera_space_to_pixel_space(camera_points, intrinsics_2)  # (b, v1, v2, h, w, 2)

    # Calculate the depth of each point
    rendered_depth = camera_points[..., 2]  # (b, v1, v2, h, w)

    # We use three conditions to determine if a point should be masked

    # Condition 1: Check if the points are in the frustum of any of the v2 views
    in_frustum_mask = (
        (points_2d[..., 0] > 0) &
        (points_2d[..., 0] < w) &
        (points_2d[..., 1] > 0) &
        (points_2d[..., 1] < h)
    )  # (b, v1, v2, h, w)
    in_frustum_mask = in_frustum_mask.any(dim=-3)  # (b, v1, h, w)

    # Condition 2: Check if the points have non-zero (i.e. valid) depth in the input view
    non_zero_depth = depth_1 > 1e-6

    # Condition 3: Check if the points have matching depth to any of the v2
    # views torch.nn.functional.grid_sample expects the input coordinates to
    # be normalized to the range [-1, 1], so we normalize first
    points_2d[..., 0] /= w
    points_2d[..., 1] /= h
    points_2d = points_2d * 2 - 1
    matching_depth = torch.ones_like(rendered_depth, dtype=torch.bool)
    for b in range(depth_1.shape[0]):
        for i in range(v1):
            for j in range(v2):
                depth = einops.rearrange(depth_2[b, j], 'h w -> 1 1 h w')
                coords = einops.rearrange(points_2d[b, i, j], 'h w c -> 1 h w c')
                sampled_depths = torch.nn.functional.grid_sample(depth, coords, align_corners=False)[0, 0]
                matching_depth[b, i, j] = torch.isclose(rendered_depth[b, i, j], sampled_depths, atol=atol)

    matching_depth = matching_depth.any(dim=-3)  # (..., v1, h, w)

    mask = in_frustum_mask & non_zero_depth & matching_depth
    return mask

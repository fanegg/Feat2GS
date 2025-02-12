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
import math
from scene.gaussian_model import GaussianModel
from utils.pose_utils import get_camera_from_tensor, quadmultiply
from utils.graphics_utils import depth_to_normal


### if use [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)

# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )
# from utils.sh_utils import eval_sh

# def render(
#     viewpoint_camera,
#     pc: GaussianModel,
#     pipe,
#     bg_color: torch.Tensor,
#     scaling_modifier=1.0,
#     override_color=None,
#     camera_pose=None,
# ):
#     """
#     Render the scene.

#     Background tensor (bg_color) must be on GPU!
#     """

#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = (
#         torch.zeros_like(
#             pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
#         )
#         + 0
#     )
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     # Set camera pose as identity. Then, we will transform the Gaussians around camera_pose
#     w2c = torch.eye(4).cuda()
#     projmatrix = (
#         w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
#     ).squeeze(0)
#     camera_pos = w2c.inverse()[3, :3]
#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         # viewmatrix=viewpoint_camera.world_view_transform,
#         # projmatrix=viewpoint_camera.full_proj_transform,
#         viewmatrix=w2c,
#         projmatrix=projmatrix,
#         sh_degree=pc.active_sh_degree,
#         # campos=viewpoint_camera.camera_center,
#         campos=camera_pos,
#         prefiltered=False,
#         debug=pipe.debug,
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     # means3D = pc.get_xyz
#     rel_w2c = get_camera_from_tensor(camera_pose)
#     # Transform mean and rot of Gaussians to camera frame
#     gaussians_xyz = pc._xyz.clone()
#     gaussians_rot = pc._rotation.clone()

#     xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
#     xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
#     gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
#     gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
#     means3D = gaussians_xyz_trans
#     means2D = screenspace_points
#     opacity = pc.get_opacity

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = gaussians_rot_trans  # pc.get_rotation

#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = pc.get_features.transpose(1, 2).view(
#                 -1, 3, (pc.max_sh_degree + 1) ** 2
#             )
#             dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
#                 pc.get_features.shape[0], 1
#             )
#             dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             shs = pc.get_features
#     else:
#         colors_precomp = override_color

#     # Rasterize visible Gaussians to image, obtain their radii (on screen).
#     rendered_image, radii = rasterizer(
#         means3D=means3D,
#         means2D=means2D,
#         shs=shs,
#         colors_precomp=colors_precomp,
#         opacities=opacity,
#         scales=scales,
#         rotations=rotations,
#         cov3D_precomp=cov3D_precomp,
#     )

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {
#         "render": rendered_image,
#         "viewspace_points": screenspace_points,
#         "visibility_filter": radii > 0,
#         "radii": radii,
#     }


### if use [gsplat](https://github.com/nerfstudio-project/gsplat)

from gsplat import rasterization

def render_gsplat(
        viewpoint_camera, 
        pc : GaussianModel, 
        pipe, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None, 
        camera_pose = None, 
        fov = None, 
        render_mode="RGB"):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if fov is None:
        FoVx = viewpoint_camera.FoVx
        FoVy = viewpoint_camera.FoVy
    else:
        FoVx = fov[0]
        FoVy = fov[1]
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation
    if override_color is not None:
        colors = override_color # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

    if camera_pose is None:
        viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
    else:
        viewmat = get_camera_from_tensor(camera_pose)
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode=render_mode,
    )

    if "D" in render_mode:
        if "+" in render_mode:
            depth_map = render_colors[..., -1:]
        else:
            depth_map = render_colors

        normals_surf = depth_to_normal(
            depth_map, torch.inverse(viewmat[None]), K[None])
        normals_surf = normals_surf * (render_alphas).detach()
        render_colors = torch.cat([render_colors, normals_surf], dim=-1)

    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)

    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": info["means2d"],
            "visibility_filter" : radii > 0,
            "radii": radii}

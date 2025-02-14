import numpy as np
import roma
import torch
import torch.nn.functional as F

from typing import Optional, Union

def rt_to_mat4(
    R: torch.Tensor, t: torch.Tensor, s: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        R (torch.Tensor): (..., 3, 3).
        t (torch.Tensor): (..., 3).
        s (torch.Tensor): (...,).

    Returns:
        torch.Tensor: (..., 4, 4)
    """
    mat34 = torch.cat([R, t[..., None]], dim=-1)
    if s is None:
        bottom = (
            mat34.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            .reshape((1,) * (mat34.dim() - 2) + (1, 4))
            .expand(mat34.shape[:-2] + (1, 4))
        )
    else:
        bottom = F.pad(1.0 / s[..., None, None], (3, 0), value=0.0)
    mat4 = torch.cat([mat34, bottom], dim=-2)
    return mat4

def get_avg_w2c(w2cs: torch.Tensor):
    c2ws = torch.linalg.inv(w2cs)
    # 1. Compute the center
    center = c2ws[:, :3, -1].mean(0)
    # 2. Compute the z axis
    z = F.normalize(c2ws[:, :3, 2].mean(0), dim=-1)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = c2ws[:, :3, 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = F.normalize(torch.cross(y_, z, dim=-1), dim=-1)  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = torch.cross(z, x, dim=-1)  # (3)
    avg_c2w = rt_to_mat4(torch.stack([x, y, z], 1), center)
    avg_w2c = torch.linalg.inv(avg_c2w)
    return avg_w2c


# def get_lookat(origins: torch.Tensor, viewdirs: torch.Tensor) -> torch.Tensor:
#     """Calculate the intersection point of multiple camera rays as the lookat point.
    
#     Use the center of camera positions as a reference point for the lookat,
#     then move forward along the average view direction by a certain distance.
#     """
#     # Calculate the center of camera positions
#     center = origins.mean(dim=0)
    
#     # Calculate average view direction
#     mean_dir = F.normalize(viewdirs.mean(dim=0), dim=-1)
    
#     # Calculate average distance to the center point
#     avg_dist = torch.norm(origins - center, dim=-1).mean()
    
#     # Move forward along the average view direction
#     lookat = center + mean_dir * avg_dist
    
#     return lookat

def get_lookat(origins: torch.Tensor, viewdirs: torch.Tensor) -> torch.Tensor:
    """Triangulate a set of rays to find a single lookat point.

    Args:
        origins (torch.Tensor): A (N, 3) array of ray origins.
        viewdirs (torch.Tensor): A (N, 3) array of ray view directions.

    Returns:
        torch.Tensor: A (3,) lookat point.
    """

    viewdirs = torch.nn.functional.normalize(viewdirs, dim=-1)
    eye = torch.eye(3, device=origins.device, dtype=origins.dtype)[None]
    # Calculate projection matrix I - rr^T
    I_min_cov = eye - (viewdirs[..., None] * viewdirs[..., None, :])
    # Compute sum of projections
    sum_proj = I_min_cov.matmul(origins[..., None]).sum(dim=-3)
    # Solve for the intersection point using least squares
    lookat = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]
    # Check NaNs.
    assert not torch.any(torch.isnan(lookat))
    return lookat


def get_lookat_w2cs(positions: torch.Tensor, lookat: torch.Tensor, up: torch.Tensor):
    """
    Args:
        positions: (N, 3) tensor of camera positions
        lookat: (3,) tensor of lookat point
        up: (3,) tensor of up vector

    Returns:
        w2cs: (N, 3, 3) tensor of world to camera rotation matrices
    """
    forward_vectors = F.normalize(lookat - positions, dim=-1)
    right_vectors = F.normalize(torch.cross(forward_vectors, up[None], dim=-1), dim=-1)
    down_vectors = F.normalize(
        torch.cross(forward_vectors, right_vectors, dim=-1), dim=-1
    )
    Rs = torch.stack([right_vectors, down_vectors, forward_vectors], dim=-1)
    w2cs = torch.linalg.inv(rt_to_mat4(Rs, positions))
    return w2cs


def get_arc_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor,
    num_frames: int,
    degree: float,
    **_,
) -> torch.Tensor:
    ref_position = torch.linalg.inv(ref_w2c)[:3, 3]
    thetas = (
        torch.sin(
            torch.linspace(0.0, torch.pi * 2.0, num_frames + 1, device=ref_w2c.device)[
                :-1
            ]
        )
        * (degree / 2.0)
        / 180.0
        * torch.pi
    )
    positions = torch.einsum(
        "nij,j->ni",
        roma.rotvec_to_rotmat(thetas[:, None] * up[None]),
        ref_position - lookat,
    )
    return get_lookat_w2cs(positions, lookat, up)


def get_lemniscate_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor,
    num_frames: int,
    degree: float,
    **_,
) -> torch.Tensor:
    ref_c2w = torch.linalg.inv(ref_w2c)
    a = torch.linalg.norm(ref_c2w[:3, 3] - lookat) * np.tan(degree / 360 * np.pi)
    # Lemniscate curve in camera space. Starting at the origin.
    thetas = (
        torch.linspace(0, 2 * torch.pi, num_frames + 1, device=ref_w2c.device)[:-1]
        + torch.pi / 2
    )

    positions = torch.stack(
        [
            a * torch.cos(thetas) / (1 + torch.sin(thetas) ** 2),
            a * torch.cos(thetas) * torch.sin(thetas) / (1 + torch.sin(thetas) ** 2),
            torch.zeros(num_frames, device=ref_w2c.device),
        ],
        dim=-1,
    )
    # Transform to world space.
    positions = torch.einsum(
        "ij,nj->ni", ref_c2w[:3], F.pad(positions, (0, 1), value=1.0)
    )
    return get_lookat_w2cs(positions, lookat, up)


def get_spiral_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor,
    num_frames: int,
    rads: Union[float, torch.Tensor],
    zrate: float,
    rots: int,
    **_,
) -> torch.Tensor:
    ref_c2w = torch.linalg.inv(ref_w2c)
    thetas = torch.linspace(
        0, 2 * torch.pi * rots, num_frames + 1, device=ref_w2c.device
    )[:-1]
    # Spiral curve in camera space. Starting at the origin.
    if isinstance(rads, torch.Tensor):
        rads = rads.reshape(-1, 3).to(ref_w2c.device)
    positions = (
        torch.stack(
            [
                torch.cos(thetas),
                -torch.sin(thetas),
                -torch.sin(thetas * zrate),
            ],
            dim=-1,
        )
        * rads
    )
    # Transform to world space.
    positions = torch.einsum(
        "ij,nj->ni", ref_c2w[:3], F.pad(positions, (0, 1), value=1.0)
    )
    return get_lookat_w2cs(positions, lookat, up)


def get_wander_w2cs(ref_w2c, focal_length, num_frames, max_disp, **_):
    device = ref_w2c.device
    c2w = np.linalg.inv(ref_w2c.detach().cpu().numpy())
    max_disp = max_disp

    max_trans = max_disp / focal_length
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = 0.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 2.0

        i_pose = np.concatenate(
            [
                np.concatenate(
                    [
                        np.eye(3),
                        np.array([x_trans, y_trans, z_trans])[:, np.newaxis],
                    ],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )

        render_pose = np.dot(ref_pose, i_pose)
        output_poses.append(render_pose)
    output_poses = torch.from_numpy(np.array(output_poses, dtype=np.float32)).to(device)
    w2cs = torch.linalg.inv(output_poses)

    return w2cs
"""Coordinate frame utilities for Aria/EFM3D.

Add a PoseTW-aware look-at helper that guarantees zero roll by aligning the
camera up-axis with the world-up vector. This preserves the LUF convention
(X-left, Y-up, Z-forward) while avoiding arbitrary yaw/roll about world Z.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from efm3d.utils.gravity import (
    GRAVITY_DIRECTION_VIO,  # [0, 0, -1]
)


def world_up_tensor(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return world up vector as tensor.

    In the EFM3D VIO convention gravity points to ``[0, 0, -1]``; world-up is
    therefore ``+Z``. This helper keeps that logic in one place.
    """

    return torch.tensor(-GRAVITY_DIRECTION_VIO, device=device, dtype=dtype)


def _view_axes_from_positions(
    cam_pos: torch.Tensor,
    look_at: torch.Tensor,
    world_up: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return camera axes (batch) for the Aria LUF camera frame.

    This is the tensor-level implementation. Use
    :func:`view_axes_from_poses` when you already operate on ``PoseTW``
    instances.
    """

    # Forward (camera z-axis) from camera to target.
    fwd = F.normalize(look_at - cam_pos, dim=-1)

    if world_up is None:
        world_up = world_up_tensor(device=cam_pos.device, dtype=cam_pos.dtype)

    # Broadcast world_up to match fwd's batch dimensions.
    while world_up.ndim < fwd.ndim:
        world_up = world_up.unsqueeze(0)

    # Choose an auxiliary up that is least aligned with forward to avoid
    # degeneracy when looking near ±Z. Fallback to world +Y when forward‖world_up.
    dot = (fwd * world_up).sum(dim=-1, keepdim=True).abs()
    alt_up = torch.tensor([0.0, 1.0, 0.0], device=cam_pos.device, dtype=cam_pos.dtype)
    while alt_up.ndim < fwd.ndim:
        alt_up = alt_up.unsqueeze(0)

    up_hint = torch.where(dot > 1.0 - 1e-3, alt_up, world_up)
    up_hint = F.normalize(up_hint + eps, dim=-1)

    # Left axis: up × forward (LUF naming; negated right).
    left = torch.cross(up_hint, fwd, dim=-1)
    left = F.normalize(left + eps, dim=-1)

    # Re-orthogonalize up to ensure left×up = forward (right-handed, LUF).
    up = torch.cross(fwd, left, dim=-1)
    up = F.normalize(up + eps, dim=-1)

    return torch.stack([left, up, fwd], dim=-1)


def view_axes_from_points(
    from_pose: PoseTW | torch.Tensor,
    look_at: PoseTW | torch.Tensor,
    world_up: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """PoseTW-aware look-at helper for LUF cameras.

    Args:
        from_pose: Camera pose (or centre tensor) in world coordinates.
        look_at: Target point (or pose whose translation is the target).
        world_up: Optional world up vector; defaults to ``+Z``.
        eps: Numerical stability term.

    Returns:
        Rotation matrices ``R_world_cam`` with shape ``(..., 3, 3)`` following
        the LUF convention (columns = left, up, forward).
    """

    cam_origin = from_pose.t if isinstance(from_pose, PoseTW) else from_pose
    target = look_at.t if isinstance(look_at, PoseTW) else look_at

    return _view_axes_from_positions(cam_pos=cam_origin, look_at=target, world_up=world_up, eps=eps)


def world_from_camera(t_world_rig: PoseTW, cam: CameraTW, frame_idx: int) -> PoseTW:
    """Compute T_world_cam from rig pose and CameraTW extrinsics."""
    return t_world_rig @ cam.T_camera_rig[frame_idx].inverse()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cam_pos = torch.tensor([[0.0, 0.0, 0.0]])  # (1, 3)
    target = torch.tensor([[0.0, 0.0, 1.0]])  # camera looks along +Z

    R_wc = view_axes_from_points(cam_pos, target)[0]  # (3, 3)
    t_wc = cam_pos[0]

    plt.show()

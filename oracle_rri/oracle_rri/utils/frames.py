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
    """Return world up vector as tensor. Default world-up: +Z (gravity up in EFM3D; gravity is [0, 0, -1]).

    Args:
        device: Desired device of the returned tensor.
        dtype: Desired data type of the returned tensor.
    Returns:
        (3,) world up vector tensor.
    """
    return torch.tensor(
        -GRAVITY_DIRECTION_VIO,
        device=device,
        dtype=dtype,
    )


def view_axes_from_points(
    cam_pos: torch.Tensor,
    look_at: torch.Tensor,
    world_up: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return camera axes (batch) for the Aria LUF camera frame.

    Args:
        cam_pos: (..., 3) camera origin in world.
        look_at: (..., 3) target point (defines forward).
        world_up: (..., 3) approximate up direction in world coordinates.
            Defaults to +Z (opposite of gravity in EFM3D's VIO convention).
        eps: Small value to avoid degeneracy when forward is close to up.

    Returns:
        (..., 3, 3) R_world_cam in SO(3) with columns:
            - [:, 0] = left    (+X_l, aligned with Aria LUF)
            - [:, 1] = up      (+Y_l, approximately aligned with world_up)
            - [:, 2] = forward (+Z_l, points from camera to look_at)

    NOTE: We build a proper right-handed rotation by enforcing left×up = forward.
    Display code may still apply a LUF→display tweak for viewers, but stored poses
    now follow the documented LUF convention directly.
    """
    # Forward (camera z-axis) from camera to target.
    fwd = F.normalize(look_at - cam_pos, dim=-1)

    if world_up is None:
        world_up = world_up_tensor(device=cam_pos.device, dtype=cam_pos.dtype)

    # Broadcast world_up to match fwd's batch dimensions.
    while world_up.ndim < fwd.ndim:
        world_up = world_up.unsqueeze(0)

    # If forward is nearly parallel to world_up, fall back to a secondary up
    # direction (world +Y) to avoid degeneracies.
    dot = (fwd * world_up).sum(dim=-1, keepdim=True).abs()
    alt_up = torch.tensor([0.0, 1.0, 0.0], device=cam_pos.device, dtype=cam_pos.dtype)
    while alt_up.ndim < fwd.ndim:
        alt_up = alt_up.unsqueeze(0)

    up_hint = torch.where(dot > 1.0 - 1e-3, alt_up, world_up)
    up_hint = F.normalize(up_hint + eps, dim=-1)

    # Left axis: world_up × forward (negated right) to follow LUF naming.
    left = torch.cross(up_hint, fwd, dim=-1)
    left = F.normalize(left + eps, dim=-1)

    # Re-orthogonalized up to ensure left×up = forward (right-handed, LUF-named).
    up = torch.cross(fwd, left, dim=-1)
    up = F.normalize(up + eps, dim=-1)

    return torch.stack([left, up, fwd], dim=-1)


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

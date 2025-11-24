import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from efm3d.utils.gravity import (
    GRAVITY_DIRECTION_VIO,  # [0, 0, -1]
    gravity_align_T_world_cam,  # type: ignore
)

LUF_TO_DISPLAY_ROT = torch.tensor(
    [
        [-1.0, 0.0, 0.0],  # flip X (left->right) for viewer
        [0.0, 1.0, 0.0],  # keep Y up to avoid upside-down frustum
        [0.0, 0.0, 1.0],
    ]
)
# NOTE: Project Aria uses LUF (left–up–forward); see docs. We flip only X to get
# a right-handed, viewer-friendly frame without inverting Y (prevents upside-down frusta).


# NOTE: Helper to align camera pose for display (used in both data and candidate plotting).
def pose_for_display(
    t_world_cam: PoseTW,
    gravity_w: torch.Tensor | None = None,
    *,
    align_gravity: bool = True,
) -> PoseTW:
    """Return a display-friendly PoseTW with gravity-up and UI roll/yaw corrections.

    Steps (mirrors data.plotting):
    1) Gravity-align so world +Z is up (undo VIO gravity tilt).
    2) Convert Aria LUF camera axes to a display frame (180° about +Z).
    3) Apply fixed rotations:
       - roll -90° about camera z to match rotated image display,
       - yaw -90° about camera y to align forward/right/down with viewer.
    This keeps frusta upright in Plotly and avoids roll (x-axis) tilt.
    """
    if align_gravity:
        if gravity_w is None:
            gravity_w = GRAVITY_DIRECTION_VIO
        t_aligned = gravity_align_T_world_cam(t_world_cam.unsqueeze(0), gravity_w=gravity_w).squeeze(0)
    else:
        t_aligned = t_world_cam
    # NOTE: previously omitted; fixes hidden RDF assumption in plotting path.
    r_luf_to_display = LUF_TO_DISPLAY_ROT.to(device=t_aligned.R.device, dtype=t_aligned.R.dtype)
    r_base = t_aligned.R @ r_luf_to_display
    # NOTE:
    #   cz, sz: roll  -90° about camera z
    #   cy, sy: display yaw; we use -90° to match efm3d/ATEK frustum orientation.
    cz, sz = (
        torch.cos(torch.tensor(-np.pi / 2, device=t_aligned.device, dtype=t_aligned.dtype)),
        torch.sin(torch.tensor(-np.pi / 2, device=t_aligned.device, dtype=t_aligned.dtype)),
    )
    cy, sy = (
        torch.cos(torch.tensor(-np.pi / 2, device=t_aligned.device, dtype=t_aligned.dtype)),
        torch.sin(torch.tensor(-np.pi / 2, device=t_aligned.device, dtype=t_aligned.dtype)),
    )
    r_z = torch.tensor(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        device=t_aligned.device,
        dtype=t_aligned.dtype,
    )
    r_y = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        device=t_aligned.device,
        dtype=t_aligned.dtype,
    )
    # First yaw about camera +Y (viewer alignment), then roll about camera +Z (image rotation).
    # NOTE: order matters – we want a pure SO(3) correction that stays compatible with
    # efm3d/ATEK visualisation.
    r_corr = r_z @ r_y
    r_disp = r_base @ r_corr
    return PoseTW.from_Rt(r_disp, t_aligned.t)


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


def world_from_rig_camera_pose(t_world_rig: PoseTW, cam: CameraTW, frame_idx: int = 0) -> PoseTW:
    """Compute T_world_cam from rig pose and CameraTW extrinsics."""
    return t_world_rig @ cam.T_camera_rig[frame_idx].inverse()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cam_pos = torch.tensor([[0.0, 0.0, 0.0]])  # (1, 3)
    target = torch.tensor([[0.0, 0.0, 1.0]])  # camera looks along +Z

    R_wc = view_axes_from_points(cam_pos, target)[0]  # (3, 3)
    t_wc = cam_pos[0]

    plt.show()

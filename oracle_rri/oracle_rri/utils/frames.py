import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from efm3d.utils.gravity import GRAVITY_DIRECTION_VIO  # [0, 0, -1]
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import plot_transform


def view_axes_from_points(
    cam_pos: torch.Tensor,
    look_at: torch.Tensor,
    world_up: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return right/up/forward axes (batch) for an RDF camera frame.

    Args:
        cam_pos: (..., 3) camera origin in world.
        look_at: (..., 3) target point (defines forward).
        world_up: (..., 3) approximate up direction in world coordinates.
            Defaults to +Z (opposite of gravity in EFM3D's VIO convention).
        eps: Small value to avoid degeneracy when forward is close to up.

    Returns:
        (..., 3, 3) rotation matrix R_world_cam with columns:
            - [:, 0] = right  (+X, R in RDF)
            - [:, 1] = up     (+Y, approximately aligned with world_up)
            - [:, 2] = forward(+Z, F in RDF, points from camera to look_at)
        The returned matrix is guaranteed to be a proper rotation
        (determinant +1) up to numerical precision.
    """
    # Forward (camera z-axis) from camera to target.
    fwd = F.normalize(look_at - cam_pos, dim=-1)

    # Default world-up: +Z (gravity up in EFM3D; gravity is [0, 0, -1]).
    if world_up is None:
        world_up = torch.tensor(
            -GRAVITY_DIRECTION_VIO,
            device=cam_pos.device,
            dtype=cam_pos.dtype,
        )

    # Broadcast world_up to match fwd's batch dimensions.
    while world_up.ndim < fwd.ndim:
        world_up = world_up.unsqueeze(0)

    # If forward is nearly parallel to world_up, fall back to a secondary up
    # direction (world +Y) to avoid degeneracies.
    dot = (fwd * world_up).sum(dim=-1, keepdim=True).abs()
    alt_up = torch.tensor(
        [0.0, 1.0, 0.0],
        device=cam_pos.device,
        dtype=cam_pos.dtype,
    )
    while alt_up.ndim < fwd.ndim:
        alt_up = alt_up.unsqueeze(0)

    up_vec = torch.where(dot > 1.0 - 1e-3, alt_up, world_up)
    up_vec = F.normalize(up_vec + eps, dim=-1)

    # Right (camera x-axis) as up x forward to obtain a right-handed frame.
    right = F.normalize(torch.cross(up_vec, fwd, dim=-1), dim=-1)
    # Re-orthogonalized up (camera y-axis) as forward x right.
    up = F.normalize(torch.cross(fwd, right, dim=-1), dim=-1)

    return torch.stack([right, up, fwd], dim=-1)


def plot_camera_frame_pytransform3d(
    r_world_cam: torch.Tensor,
    t_world_cam: torch.Tensor,
    axis_length: float = 0.2,
    ax=None,
):
    """Plot a camera frame given r_world_cam and t_world_cam.

    Args:
        r_world_cam: (3, 3) rotation matrix from camera to world (columns are
            camera axes expressed in world).
        t_world_cam: (3,) camera origin in world coordinates.
        axis_length: Length of the plotted axes.
        ax: Optional existing 3D axis; if None, a new one is created.

    Returns:
        Matplotlib 3D axis with the camera frame drawn (X=red, Y=green, Z=blue).
    """
    r = r_world_cam.detach().cpu().numpy()
    t = t_world_cam.detach().cpu().numpy()

    # Build 4x4 homogeneous transform from camera to world
    a2b = np.eye(4)
    a2b[:3, :3] = r
    a2b[:3, 3] = t

    if ax is None:
        ax = make_3d_axis(axis_length)

    plot_transform(ax=ax, A2B=a2b)
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_zlabel("Z (world)")
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cam_pos = torch.tensor([[0.0, 0.0, 0.0]])  # (1, 3)
    target = torch.tensor([[0.0, 0.0, 1.0]])  # camera looks along +Z

    R_wc = view_axes_from_points(cam_pos, target)[0]  # (3, 3)
    t_wc = cam_pos[0]

    ax = plot_camera_frame_pytransform3d(R_wc, t_wc, axis_length=0.5)
    plt.show()

"""Plotting utilities for the Streamlit data view.

Design goals:
- Single, deterministic code path (no debug toggles).
- efm3d/ATEK conventions: camera RDF (x right, y down, z forward), world Z-up (gravity = [0, 0, -g]).
- Pose: T_world_cam = T_world_rig @ T_camera_rig.inverse() (efm3d render_frustum convention), then gravity-align.
- Visualization: roll -90° about camera z to match the -90° image display.
- Frustum: distortion-aware via CameraTW.unproject of image corners (valid-radius square like efm3d).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from efm3d.utils.gravity import GRAVITY_DIRECTION_VIO, gravity_align_T_world_cam
from plotly.subplots import make_subplots

from oracle_rri.data import EfmSnippetView


def _frustum_segments(cam: CameraTW, pose_world_cam: PoseTW, scale: float = 1.0) -> list[np.ndarray]:
    """Return frustum wireframe segments in world frame using CameraTW.unproject."""
    c = cam.c.squeeze(0)
    # mimic efm3d render_frustum: inscribed square using valid_radius * sqrt(0.5)
    rs = cam.valid_radius.squeeze(0) * 0.7071
    corners_px = torch.stack(
        [
            c + torch.tensor([-rs[0], -rs[1]], device=cam.device, dtype=cam.dtype),
            c + torch.tensor([-rs[0], rs[1]], device=cam.device, dtype=cam.dtype),
            c + torch.tensor([rs[0], rs[1]], device=cam.device, dtype=cam.dtype),
            c + torch.tensor([rs[0], -rs[1]], device=cam.device, dtype=cam.dtype),
        ],
        dim=0,
    ).unsqueeze(0)  # 1 x 4 x 2

    rays_cam, valid = cam.unproject(corners_px)  # B x N x 3
    rays_cam = rays_cam.squeeze(0)
    valid = valid.squeeze(0)
    rays_cam = torch.where(valid.unsqueeze(-1), rays_cam, torch.zeros_like(rays_cam))
    rays_cam = torch.nn.functional.normalize(rays_cam, dim=-1, eps=1e-6)
    frustum_cam = rays_cam * scale  # 4 x 3

    frustum_world = pose_world_cam.transform(frustum_cam)  # 4 x 3
    center = pose_world_cam.t

    frustum_np = frustum_world.detach().cpu().numpy()
    center_np = center.detach().cpu().numpy()

    segments = []
    for i in range(4):
        segments.append(np.vstack([center_np, frustum_np[i], frustum_np[(i + 1) % 4]]))

    # small top triangle for orientation cue
    up = frustum_np[0] - frustum_np[1]
    top_pts = np.array(
        [
            frustum_np[0] + 0.1 * up,
            0.5 * (frustum_np[0] + frustum_np[3]) + 0.5 * up,
            frustum_np[3] + 0.1 * up,
        ]
    )
    segments.append(top_pts)
    return segments


def _roll_about_z(pose: PoseTW, angle_rad: float) -> PoseTW:
    """Roll camera around its own z-axis (keeps translation)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    r_roll = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=pose.R.device, dtype=pose.R.dtype)
    r_new = pose.R @ r_roll
    return PoseTW.from_Rt(r_new, pose.t)


def _to_uint8_image(img: torch.Tensor, *, rotate_k: int = 0) -> np.ndarray:
    """Convert tensor image (C,H,W or H,W) to uint8 HWC."""
    arr = img.detach().cpu()
    if arr.ndim == 2:
        arr = arr.unsqueeze(0)
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    arr = arr.permute(1, 2, 0).clamp(0, 1).mul(255).to(torch.uint8).numpy()
    if rotate_k:
        arr = np.rot90(arr, k=rotate_k)
    return arr


def plot_frames(sample: EfmSnippetView) -> go.Figure:
    """First RGB/SLAM frames side-by-side."""
    cam_rgb = sample.camera_rgb
    cam_l = sample.camera_slam_left
    cam_r = sample.camera_slam_right

    rotate_k = -1  # match legacy UI orientation
    rgb_img = _to_uint8_image(cam_rgb.images[0], rotate_k=rotate_k)
    slam_l_img = _to_uint8_image(cam_l.images[0], rotate_k=rotate_k)
    slam_r_img = _to_uint8_image(cam_r.images[0], rotate_k=rotate_k)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("RGB", "SLAM-L", "SLAM-R"),
        specs=[[{"type": "image"}] * 3],
    )
    fig.add_trace(go.Image(z=rgb_img), row=1, col=1)
    fig.add_trace(go.Image(z=slam_l_img), row=1, col=2)
    fig.add_trace(go.Image(z=slam_r_img), row=1, col=3)
    fig.update_layout(height=500, width=1200, title_text="Camera views (frame 0)")
    return fig


def plot_trajectory(sample: EfmSnippetView) -> go.Figure:
    """Mesh + semidense + trajectory + single camera frustum."""
    mesh = sample.mesh
    traj = sample.trajectory
    sem = sample.semidense

    cam = sample.camera_rgb
    cam_ts = np.atleast_1d(cam.time_ns.cpu().numpy())
    traj_ts = np.atleast_1d(traj.time_ns.cpu().numpy())
    if cam_ts.size == 0 or traj_ts.size == 0:
        return go.Figure()
    frame_idx = 0
    traj_idx = int(np.argmin(np.abs(traj_ts - cam_ts[frame_idx])))

    # Pose: world←camera = world←rig @ rig←camera (inverse to match efm3d render_frustum).
    t_world_rig = traj.t_world_rig[traj_idx]
    t_cam_rig = cam.calib.T_camera_rig[frame_idx]
    t_world_cam = t_world_rig @ t_cam_rig.inverse()

    # Gravity-align to keep world Z-up consistent with efm3d.
    t_world_cam = gravity_align_T_world_cam(t_world_cam.unsqueeze(0), gravity_w=GRAVITY_DIRECTION_VIO).squeeze(0)

    # Roll -90° about camera z to match displayed images.
    t_world_cam_vis = _roll_about_z(t_world_cam, angle_rad=-np.pi / 2)

    cam_center = t_world_cam_vis.t.detach().cpu().numpy()
    traj_positions = traj.t_world_rig.t.detach().cpu().numpy()
    traj_x, traj_y, traj_z = traj_positions.T

    # Semidense points (world frame).
    pts_np = np.zeros((0, 3))
    if sem is not None and hasattr(sem, "points_world"):
        points = sem.points_world[frame_idx, : int(sem.lengths[frame_idx].item())]
        if points.ndim == 1:
            points = points.unsqueeze(0)
        if points.shape[0] > 20000:
            idx = torch.randperm(points.shape[0])[:20000]
            points = points[idx]
        pts_np = points.detach().cpu().numpy()

    # Camera axes (rolled).
    cam_axes = (
        t_world_cam_vis.rotate(torch.eye(3, device=t_world_cam_vis.device, dtype=t_world_cam_vis.dtype))
        .detach()
        .cpu()
        .numpy()
    )

    fig = go.Figure()

    if mesh is not None:
        fig.add_trace(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color="lightgray",
                opacity=0.25,
                name="GT Mesh",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=traj_x,
            y=traj_y,
            z=traj_z,
            mode="lines+markers",
            marker={"size": 3, "color": "black"},
            line={"color": "black", "width": 4},
            name="Trajectory",
        )
    )

    if pts_np.size > 0:
        fig.add_trace(
            go.Scatter3d(
                x=pts_np[:, 0],
                y=pts_np[:, 1],
                z=pts_np[:, 2],
                mode="markers",
                marker={
                    "size": 2,
                    "color": pts_np[:, 2],
                    "colorscale": "Viridis",
                    "opacity": 0.5,
                },
                name="Semidense points",
            )
        )

    # Frustum
    frustum_segments = _frustum_segments(
        cam.calib[frame_idx],
        t_world_cam_vis,
        virtual_image_distance=1.0,
    )
    for idx, seg in enumerate(frustum_segments):
        fig.add_trace(
            go.Scatter3d(
                x=seg[:, 0],
                y=seg[:, 1],
                z=seg[:, 2],
                mode="lines",
                line={"color": "red", "width": 4},
                name="Frustum" if idx == 0 else None,
                showlegend=idx == 0,
            )
        )

    # Axes
    axis_colors = ["red", "green", "blue"]
    for axis, color in enumerate(axis_colors):
        axis_end = cam_center + cam_axes[axis] * 0.4
        fig.add_trace(
            go.Scatter3d(
                x=[cam_center[0], axis_end[0]],
                y=[cam_center[1], axis_end[1]],
                z=[cam_center[2], axis_end[2]],
                mode="lines",
                line={"color": color, "width": 8},
                name="Camera axes" if axis == 0 else None,
                showlegend=axis == 0,
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[cam_center[0]],
            y=[cam_center[1]],
            z=[cam_center[2]],
            mode="markers",
            marker={"size": 10, "color": "red", "symbol": "diamond"},
            name="Camera center",
        )
    )

    # Scene bounds
    bbox = np.vstack(
        [
            pts_np,
            cam_center.reshape(1, 3),
            traj_positions,
            mesh.vertices if mesh is not None else np.zeros((0, 3)),
        ]
    )
    finite_mask = np.isfinite(bbox).all(axis=1)
    bbox = bbox[finite_mask]
    if bbox.shape[0] == 0:
        bbox = np.array([[0.0, 0.0, 0.0]])

    vmin, vmax = bbox.min(axis=0), bbox.max(axis=0)
    max_extent = (vmax - vmin).max()
    padding = max(0.5, 0.1 * max_extent)
    scene_ranges = {
        "xaxis": {"title": "X (m)", "range": [vmin[0] - padding, vmax[0] + padding]},
        "yaxis": {"title": "Y (m)", "range": [vmin[1] - padding, vmax[1] + padding]},
        "zaxis": {"title": "Z (m)", "range": [vmin[2] - padding, vmax[2] + padding]},
    }

    fig.update_layout(
        title="Mesh + semidense + trajectory + camera frustum",
        scene=dict(aspectmode="data", **scene_ranges),
        height=900,
    )
    return fig


if __name__ == "__main__":
    pass

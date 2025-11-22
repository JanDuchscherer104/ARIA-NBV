"""Plotting utilities for the Streamlit data view.

Conventions (efm3d / ATEK):
- Camera frame is RDF (x right, y down, z forward); world is Z-up (gravity = [0,0,-g]).
- Pose: T_world_cam = T_world_rig @ T_camera_rig.inverse() (efm3d render_frustum).
- Gravity-align to keep +Z up, then rebase orientation so viewer up = -camera x
  (efm3d viz convention to undo Aria's 90° roll and match displayed frames).
- Frustum is distortion-aware via CameraTW.unproject of an inscribed valid-radius
  square (same as efm3d.render_frustum).
- Images are rotated -90° for display; frustum/axes follow the rebased pose.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from efm3d.utils.gravity import GRAVITY_DIRECTION_VIO, gravity_align_T_world_cam
from plotly.subplots import make_subplots

from oracle_rri.data import EfmCameraView, EfmSnippetView, EfmTrajectoryView
from oracle_rri.utils import Console

console = Console.with_prefix("plotting")


def _frustum_segments(cam: CameraTW, pose_world_cam: PoseTW, scale: float = 1.0) -> list[np.ndarray]:
    """Return frustum wireframe segments in world frame using CameraTW.unproject."""
    c = cam.c.squeeze(0)
    rs = cam.valid_radius.squeeze(0) * 0.7071  # inscribed square
    corners_px = torch.stack(
        [
            c + torch.tensor([-rs[0], -rs[1]], device=cam.device, dtype=cam.dtype),
            c + torch.tensor([-rs[0], rs[1]], device=cam.device, dtype=cam.dtype),
            c + torch.tensor([rs[0], rs[1]], device=cam.device, dtype=cam.dtype),
            c + torch.tensor([rs[0], -rs[1]], device=cam.device, dtype=cam.dtype),
        ],
        dim=0,
    ).unsqueeze(0)  # 1 x 4 x 2

    rays_cam, valid = cam.unproject(corners_px)  # 1 x 4 x 3
    rays_cam = rays_cam.squeeze(0)
    valid = valid.squeeze(0)
    rays_cam = torch.where(valid.unsqueeze(-1), rays_cam, torch.zeros_like(rays_cam))
    rays_cam = torch.nn.functional.normalize(rays_cam, dim=-1, eps=1e-6)
    frustum_cam = rays_cam * scale

    frustum_world = pose_world_cam.transform(frustum_cam)  # 4 x 3
    center = pose_world_cam.t

    frustum_np = frustum_world.detach().cpu().numpy()
    center_np = center.detach().cpu().numpy()

    segments = []
    for i in range(4):
        segments.append(np.vstack([center_np, frustum_np[i], frustum_np[(i + 1) % 4]]))
    # small top triangle cue
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


def _bbox_edges(min_pt: np.ndarray, max_pt: np.ndarray) -> list[np.ndarray]:
    corners = np.array(
        [
            [min_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]],
        ]
    )
    edges_idx = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return [corners[list(pair)] for pair in edges_idx]


def _mesh_to_plotly(mesh, *, double_sided: bool = False) -> list[go.Mesh3d]:
    """Convert a Trimesh to one or two Plotly Mesh3d traces (inner + outer)."""

    vertices = mesh.vertices
    faces = mesh.faces
    traces: list[go.Mesh3d] = []

    traces.append(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="lightgray",
            opacity=0.35,
            flatshading=True,
            lighting={"ambient": 1.0, "diffuse": 0.3, "specular": 0.0, "fresnel": 0.0, "roughness": 1.0},
            name="GT Mesh",
            hoverinfo="skip",
            showscale=False,
        )
    )

    if double_sided:
        eps = 0.003
        v_normals = mesh.vertex_normals
        inner_vertices = vertices - eps * v_normals
        faces_rev = faces[:, [0, 2, 1]]
        traces.append(
            go.Mesh3d(
                x=inner_vertices[:, 0],
                y=inner_vertices[:, 1],
                z=inner_vertices[:, 2],
                i=faces_rev[:, 0],
                j=faces_rev[:, 1],
                k=faces_rev[:, 2],
                color="#c0c0c0",
                opacity=0.45,
                flatshading=True,
                lighting={"ambient": 1.0, "diffuse": 0.0, "specular": 0.0, "fresnel": 0.0, "roughness": 1.0},
                name="GT Mesh (inner)",
                hoverinfo="skip",
                showscale=False,
            )
        )

    return traces


def _aligned_pose_world_cam(cam: EfmCameraView, traj: EfmTrajectoryView, frame_idx: int) -> PoseTW:
    """Align camera frame to nearest rig pose, gravity-align, and apply display corrections."""

    cam_ts = np.atleast_1d(cam.time_ns.cpu().numpy())
    traj_ts = np.atleast_1d(traj.time_ns.cpu().numpy())
    if cam_ts.size == 0 or traj_ts.size == 0:
        return PoseTW.from_Rt(torch.eye(3), torch.zeros(3))
    traj_idx = int(np.argmin(np.abs(traj_ts - cam_ts[frame_idx])))

    t_world_rig = traj.t_world_rig[traj_idx]
    t_cam_rig = cam.calib.T_camera_rig[frame_idx]
    t_world_cam = t_world_rig @ t_cam_rig.inverse()

    t_world_cam = gravity_align_T_world_cam(t_world_cam.unsqueeze(0), gravity_w=GRAVITY_DIRECTION_VIO).squeeze(0)

    cz, sz = np.cos(-np.pi / 2), np.sin(-np.pi / 2)
    cy, sy = np.cos(np.pi / 2), np.sin(np.pi / 2)
    r_z = torch.tensor(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], device=t_world_cam.device, dtype=t_world_cam.dtype
    )
    r_y = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], device=t_world_cam.device, dtype=t_world_cam.dtype
    )
    r_corr = r_z @ r_y
    r_disp = t_world_cam.R @ r_corr
    return PoseTW.from_Rt(r_disp, t_world_cam.t)


class TrajectoryPlotBuilder:
    """Composable builder for the trajectory + semidense + mesh Plotly figure."""

    def __init__(self, *, scene_ranges: dict, title: str, height: int = 900):
        self.fig = go.Figure()
        self.scene_ranges = scene_ranges
        self.title = title
        self.height = height

    def add_mesh(self, mesh, *, double_sided: bool = False) -> "TrajectoryPlotBuilder":
        if mesh is None:
            return self
        for trace in _mesh_to_plotly(mesh, double_sided=double_sided):
            self.fig.add_trace(trace)
        return self

    def add_semidense(self, pts_np: np.ndarray, *, name: str = "Semidense points") -> "TrajectoryPlotBuilder":
        if pts_np.size == 0:
            return self
        self.fig.add_trace(
            go.Scatter3d(
                x=pts_np[:, 0],
                y=pts_np[:, 1],
                z=pts_np[:, 2],
                mode="markers",
                marker={"size": 2, "color": pts_np[:, 2], "colorscale": "Viridis", "opacity": 0.5},
                name=name,
            )
        )
        return self

    def add_trajectory(
        self,
        traj_positions: np.ndarray,
        *,
        mark_first_last: bool = True,
        first_color: str = "green",
        last_color: str = "red",
    ) -> "TrajectoryPlotBuilder":
        traj_x, traj_y, traj_z = traj_positions.T
        self.fig.add_trace(
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
        if mark_first_last and traj_positions.shape[0] > 0:
            first = traj_positions[0]
            last = traj_positions[-1]
            self.fig.add_trace(
                go.Scatter3d(
                    x=[first[0]],
                    y=[first[1]],
                    z=[first[2]],
                    mode="markers",
                    marker={"size": 2, "color": first_color, "symbol": "diamond"},
                    name="Start",
                )
            )
            if traj_positions.shape[0] > 1:
                self.fig.add_trace(
                    go.Scatter3d(
                        x=[last[0]],
                        y=[last[1]],
                        z=[last[2]],
                        mode="markers",
                        marker={"size": 2, "color": last_color, "symbol": "x"},
                        name="Final",
                    )
                )
        return self

    def add_frustum(
        self, cam: CameraTW, pose_world_cam: PoseTW, *, scale: float = 1.0, name: str = "Frustum"
    ) -> "TrajectoryPlotBuilder":
        frustum_segments = _frustum_segments(cam, pose_world_cam, scale=scale)
        for idx, seg in enumerate(frustum_segments):
            self.fig.add_trace(
                go.Scatter3d(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    z=seg[:, 2],
                    mode="lines",
                    line={"color": "red", "width": 4},
                    name=name if idx == 0 else None,
                    showlegend=idx == 0,
                )
            )
        return self

    def add_camera_axes(self, cam_center: np.ndarray, cam_axes: np.ndarray) -> "TrajectoryPlotBuilder":
        axis_colors = ["red", "green", "blue"]
        for axis, color in enumerate(axis_colors):
            axis_end = cam_center + cam_axes[axis] * 0.4
            self.fig.add_trace(
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
        return self

    def add_camera_center(
        self, cam_center: np.ndarray, *, color: str = "red", symbol: str = "diamond"
    ) -> "TrajectoryPlotBuilder":
        self.fig.add_trace(
            go.Scatter3d(
                x=[cam_center[0]],
                y=[cam_center[1]],
                z=[cam_center[2]],
                mode="markers",
                marker={"size": 10, "color": color, "symbol": symbol},
                name="Camera center",
            )
        )
        return self

    def add_bounds_box(
        self,
        min_pt: np.ndarray,
        max_pt: np.ndarray,
        *,
        name: str,
        color: str = "gray",
        dash: str = "dash",
        width: int = 2,
    ) -> "TrajectoryPlotBuilder":
        for idx, seg in enumerate(_bbox_edges(min_pt, max_pt)):
            self.fig.add_trace(
                go.Scatter3d(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    z=seg[:, 2],
                    mode="lines",
                    line={"color": color, "width": width, "dash": dash},
                    name=name if idx == 0 else None,
                    showlegend=idx == 0,
                    hoverinfo="skip",
                )
            )
        return self

    def finalize(self) -> go.Figure:
        self.fig.update_layout(title=self.title, scene=dict(aspectmode="data", **self.scene_ranges), height=self.height)
        return self.fig


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

    rotate_k = -1  # legacy UI orientation
    imgs = [
        _to_uint8_image(cam_rgb.images[0], rotate_k=rotate_k),
        _to_uint8_image(cam_l.images[0], rotate_k=rotate_k),
        _to_uint8_image(cam_r.images[0], rotate_k=rotate_k),
    ]

    fig = make_subplots(rows=1, cols=3, subplot_titles=("RGB", "SLAM-L", "SLAM-R"), specs=[[{"type": "image"}] * 3])
    for idx, img in enumerate(imgs, start=1):
        fig.add_trace(go.Image(z=img), row=1, col=idx)
    fig.update_layout(height=500, width=1200, title_text="Camera views (frame 0)")
    return fig


def plot_first_last_frames(sample: EfmSnippetView) -> go.Figure:
    """First and final RGB/SLAM frames in a 2x3 grid."""

    cams = [sample.camera_rgb, sample.camera_slam_left, sample.camera_slam_right]
    titles = ("RGB", "SLAM-L", "SLAM-R")
    n_frames = cams[0].images.shape[0]
    last_idx = max(n_frames - 1, 0)

    rotate_k = -1
    subplot_titles = [f"First {t}" for t in titles]
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=subplot_titles,
        specs=[[{"type": "image"}] * 3, [{"type": "image"}] * 3],
    )
    for col, cam in enumerate(cams, start=1):
        fig.add_trace(go.Image(z=_to_uint8_image(cam.images[0], rotate_k=rotate_k)), row=1, col=col)
        fig.add_trace(go.Image(z=_to_uint8_image(cam.images[last_idx], rotate_k=rotate_k)), row=2, col=col)

    fig.update_layout(height=900, width=1200, title_text="First and final frames (rotated for display)")
    return fig


def plot_trajectory(
    sample: EfmSnippetView,
    camera: str = "rgb",
    show_semidense: bool = True,
    max_sem_points: int | None = 20000,
    pc_from_last_only: bool = True,
    show_scene_bounds: bool = True,
    show_crop_bounds: bool = False,
    crop_aabb: tuple[np.ndarray, np.ndarray] | None = None,
    show_frustum: bool = True,
    frustum_frame_indices: list[int] | None = None,
    frustum_scale: float = 1.0,
    mark_first_last: bool = True,
    double_sided_mesh: bool = False,
) -> go.Figure:
    """Mesh + semidense + trajectory + optional frusta/bounds."""

    mesh = sample.mesh
    traj = sample.trajectory
    sem = sample.semidense

    cam_map = {
        "rgb": sample.camera_rgb,
        "slam-l": sample.camera_slam_left,
        "slam-r": sample.camera_slam_right,
    }
    cam = cam_map.get(camera, sample.camera_rgb)

    if frustum_frame_indices is None or len(frustum_frame_indices) == 0:
        frustum_frame_indices = [0]

    clamped_indices = [max(0, min(idx, cam.time_ns.shape[0] - 1)) for idx in frustum_frame_indices]
    frustum_poses = [_aligned_pose_world_cam(cam, traj, idx) for idx in clamped_indices]

    traj_positions = traj.t_world_rig.t.detach().cpu().numpy()
    pts_np = np.zeros((0, 3), dtype=np.float32)
    if show_semidense and sem is not None:
        pts_np = (
            sem.last_frame_points_np(max_sem_points) if pc_from_last_only else sem.collapse_points_np(max_sem_points)
        )

    cam_centers = np.stack([p.t.detach().cpu().numpy() for p in frustum_poses], axis=0)
    cam_axes = [p.R.detach().cpu().numpy() for p in frustum_poses]

    bbox = np.vstack(
        [
            pts_np,
            cam_centers.reshape(-1, 3),
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

    scene_min = scene_max = None
    if sem is not None and hasattr(sem, "volume_min") and hasattr(sem, "volume_max"):
        scene_min = sem.volume_min.detach().cpu().numpy()
        scene_max = sem.volume_max.detach().cpu().numpy()

    builder = TrajectoryPlotBuilder(title="Mesh + semidense + trajectory + camera frustum", scene_ranges=scene_ranges)
    builder.add_mesh(mesh, double_sided=double_sided_mesh).add_trajectory(
        traj_positions, mark_first_last=mark_first_last
    )

    if pts_np.size > 0:
        builder.add_semidense(pts_np)

    if show_frustum:
        for pose, idx in zip(frustum_poses, clamped_indices, strict=False):
            builder.add_frustum(cam.calib[idx], pose, scale=frustum_scale)
        builder.add_camera_axes(cam_centers[0], cam_axes[0])
        builder.add_camera_center(cam_centers[0])

    if show_scene_bounds:
        if scene_min is not None and scene_max is not None:
            builder.add_bounds_box(scene_min, scene_max, name="Scene bounds", color="gray", dash="dash", width=2)
        else:
            builder.add_bounds_box(vmin, vmax, name="Scene bounds", color="gray", dash="dash", width=2)

    if show_crop_bounds and crop_aabb is not None:
        builder.add_bounds_box(crop_aabb[0], crop_aabb[1], name="Crop bounds", color="orange", dash="solid", width=3)
    elif show_crop_bounds and mesh is not None:
        mesh_min, mesh_max = mesh.bounds
        builder.add_bounds_box(mesh_min, mesh_max, name="Crop bounds", color="orange", dash="solid", width=3)

    return builder.finalize()


def plot_crop_box(sample: EfmSnippetView, margin: float = 0.0, color: str = "orange") -> go.Figure:
    """Visualize the semidense AABB (with optional margin) and mesh/trajectory context."""

    # Prefer computing bounds from finite semidense points to avoid oversized volumes.
    sem = sample.semidense
    if sem is None or not hasattr(sem, "points_world"):
        return go.Figure()
    lengths = sem.lengths
    frame_idx = int(torch.argmax(lengths).item())
    n_valid = int(lengths[frame_idx].item())
    points = sem.points_world[frame_idx, :n_valid]
    finite = torch.isfinite(points).all(dim=-1)
    points = points[finite]
    if points.shape[0] == 0:
        return go.Figure()
    pts_np = points.detach().cpu().numpy()
    # Outlier-robust bounds (1st-99th percentile) to avoid a few stray points inflating the box.
    lo_np = np.nanpercentile(pts_np, 1, axis=0) - margin
    hi_np = np.nanpercentile(pts_np, 99, axis=0) + margin
    console.dbg(
        f"crop box from semidense frame {frame_idx}: n={pts_np.shape[0]}, lo={lo_np}, hi={hi_np}, margin={margin}"
    )

    def _edges(a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
        corners = np.array(
            [
                [a[0], a[1], a[2]],
                [b[0], a[1], a[2]],
                [b[0], b[1], a[2]],
                [a[0], b[1], a[2]],
                [a[0], a[1], b[2]],
                [b[0], a[1], b[2]],
                [b[0], b[1], b[2]],
                [a[0], b[1], b[2]],
            ]
        )
        edges_idx = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        return [corners[list(pair)] for pair in edges_idx]

    edges = _edges(lo_np, hi_np)
    fig = go.Figure()

    for idx, seg in enumerate(edges):
        fig.add_trace(
            go.Scatter3d(
                x=seg[:, 0],
                y=seg[:, 1],
                z=seg[:, 2],
                mode="lines",
                line={"color": color, "width": 6 if idx < 4 else 3},
                name="Crop AABB" if idx == 0 else None,
                showlegend=idx == 0,
            )
        )

    if sample.mesh is not None:
        fig.add_trace(
            go.Mesh3d(
                x=sample.mesh.vertices[:, 0],
                y=sample.mesh.vertices[:, 1],
                z=sample.mesh.vertices[:, 2],
                i=sample.mesh.faces[:, 0],
                j=sample.mesh.faces[:, 1],
                k=sample.mesh.faces[:, 2],
                color="lightgray",
                opacity=0.2,
                name="Mesh",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    traj_positions = sample.trajectory.t_world_rig.matrix3x4[..., :3, 3].detach().cpu().numpy()
    fig.add_trace(
        go.Scatter3d(
            x=traj_positions[:, 0],
            y=traj_positions[:, 1],
            z=traj_positions[:, 2],
            mode="lines+markers",
            marker={"size": 3, "color": "black"},
            line={"color": "black", "width": 3},
            name="Trajectory",
        )
    )

    pts = sample.semidense.points_world
    n_valid = int(sample.semidense.lengths[0].item()) if hasattr(sample.semidense, "lengths") else 0
    if n_valid > 0:
        pts = pts[0, :n_valid]
        finite = torch.isfinite(pts).all(dim=-1)
        pts = pts[finite]
        if pts.shape[0] > 5000:
            pts = pts[torch.randperm(pts.shape[0])[:5000]]
        pts_np = pts.detach().cpu().numpy()
        fig.add_trace(
            go.Scatter3d(
                x=pts_np[:, 0],
                y=pts_np[:, 1],
                z=pts_np[:, 2],
                mode="markers",
                marker={"size": 2, "opacity": 0.5, "color": "purple"},
                name="Semidense (sampled)",
            )
        )

    ranges = {
        "xaxis": {"title": "X (m)", "range": [lo_np[0], hi_np[0]]},
        "yaxis": {"title": "Y (m)", "range": [lo_np[1], hi_np[1]]},
        "zaxis": {"title": "Z (m)", "range": [lo_np[2], hi_np[2]]},
    }
    fig.update_layout(
        title=f"Crop AABB (margin={margin:.2f} m)",
        scene=dict(aspectmode="data", **ranges),
        height=700,
    )
    return fig


def crop_aabb_from_semidense(sample: EfmSnippetView, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (lo, hi) crop bounds from semidense points; robust to outliers."""

    sem = sample.semidense
    if sem is None or not hasattr(sem, "points_world"):
        return None
    lengths = sem.lengths
    if lengths.numel() == 0:
        return None
    frame_idx = int(torch.argmax(lengths).item())
    n_valid = int(lengths[frame_idx].item())
    if n_valid == 0:
        return None
    points = sem.points_world[frame_idx, :n_valid]
    finite = torch.isfinite(points).all(dim=-1)
    points = points[finite]
    if points.shape[0] == 0:
        return None
    pts_np = points.detach().cpu().numpy()
    lo_np = np.nanpercentile(pts_np, 1, axis=0) - margin
    hi_np = np.nanpercentile(pts_np, 99, axis=0) + margin
    return lo_np, hi_np


if __name__ == "__main__":
    from oracle_rri.data import AseEfmDatasetConfig
    from oracle_rri.utils import Console

    debug = True
    verbose = True
    console = Console.with_prefix("tst_import").set_verbose(True).set_debug(debug)

    ds_cfg = AseEfmDatasetConfig(
        scene_ids=["81283"],
        mesh_simplify_ratio=0.01,
        load_meshes=True,
        require_mesh=True,
        verbose=verbose,
        is_debug=debug,
    )
    dataset = ds_cfg.setup_target()
    sample = next(iter(dataset))

    plot_trajectory(sample, pc_from_last_only=False).show()
    plot_crop_box(sample, margin=0.2).show()

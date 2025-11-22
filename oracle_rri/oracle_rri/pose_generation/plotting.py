"""Plotting helpers for candidate sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go  # type: ignore[import]
import torch
import trimesh
from efm3d.aria.pose import PoseTW

from oracle_rri.configs import PathConfig
from oracle_rri.data import AseEfmDatasetConfig
from oracle_rri.data.plotting import SnippetPlotBuilder, _frustum_segments, _mesh_to_plotly
from oracle_rri.pose_generation import CandidateViewGenerator, CandidateViewGeneratorConfig
from oracle_rri.utils import Console
from oracle_rri.utils.frames import pose_for_display

console = Console.with_prefix("pose_plotting")


def _ensure_plot_dir() -> Path:
    out = PathConfig().data_root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


# NOTE: data.plotting doesn't expose pose extraction; reintroduce a local robust helper.
def _pose_positions(poses: torch.Tensor | "PoseTW") -> np.ndarray:
    """Extract Nx3 positions from PoseTW or (N,12)/(N,3,4)/(N,3) tensors."""

    if hasattr(poses, "matrix3x4"):
        arr = poses.matrix3x4
    elif hasattr(poses, "_data"):
        arr = poses._data  # PoseTW stores raw tensor as _data
    else:
        arr = poses

    if arr.dim() == 2 and arr.shape == torch.Size([3, 4]):
        pos = arr[:3, 3].unsqueeze(0)
    elif arr.dim() == 2 and arr.shape[1] == 12:
        pos = arr.view(-1, 3, 4)[..., :3, 3]
    elif arr.dim() == 3 and arr.shape[1:] == (3, 4):
        pos = arr[..., :3, 3]
    elif arr.dim() == 2 and arr.shape[1] == 3:
        pos = arr
    else:
        # NOTE: fallback to last column if shape unexpected
        pos = arr[..., :3]
    return pos.detach().cpu().numpy()


def camera_intrinsics(cam, frame_idx: int = 0) -> np.ndarray:
    """Return 3x3 pinhole intrinsics; accepts EfmCameraView or CameraTW."""
    cam_tw = cam.calib if hasattr(cam, "calib") else cam
    data = cam_tw.tensor()
    fx, fy = data[frame_idx, cam_tw.F_IND].tolist()
    cx, cy = data[frame_idx, cam_tw.C_IND].tolist()
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


# NOTE: still using pinhole proxy intrinsics; fisheye distortion ignored (known limitation).
def camera_wireframe_segments(
    intrinsics: np.ndarray,
    cam2world: np.ndarray,
    sensor_size: tuple[int, int],
    virtual_image_distance: float = 0.4,
) -> list[np.ndarray]:
    """Return segment polylines for a simple pyramidal camera frustum."""
    focal = float(np.mean((intrinsics[0, 0], intrinsics[1, 1])))
    w, h = sensor_size
    sensor_corners = np.array(
        [
            [0, 0, focal],
            [0, h, focal],
            [w, h, focal],
            [w, 0, focal],
        ]
    )
    sensor_corners[:, 0] -= intrinsics[0, 2]
    sensor_corners[:, 1] -= intrinsics[1, 2]
    sensor_corners *= virtual_image_distance / focal

    up = sensor_corners[0] - sensor_corners[1]
    top_corners = np.array(
        [
            sensor_corners[0] + 0.1 * up,
            0.5 * (sensor_corners[0] + sensor_corners[3]) + 0.5 * up,
            sensor_corners[3] + 0.1 * up,
            sensor_corners[0] + 0.1 * up,
        ]
    )

    def transform(points: np.ndarray) -> np.ndarray:
        pts_h = np.hstack([points, np.ones((len(points), 1))])
        return (cam2world @ pts_h.T).T[:, :3]

    sensor_world = transform(sensor_corners)
    top_world = transform(top_corners)
    center = cam2world[:3, 3]

    segments: list[np.ndarray] = []
    for i in range(4):
        segments.append(np.vstack([center, sensor_world[i], sensor_world[(i + 1) % 4]]))
    segments.append(top_world)
    return segments


def plot_candidates(
    poses: torch.Tensor | PoseTW,
    mesh: trimesh.Trimesh | None,
    title: str = "Candidate positions",
    path: Path | None = None,
    center: np.ndarray | None = None,
) -> go.Figure:
    """Plot candidate camera centres with optional mesh overlay."""
    pos = _pose_positions(poses)
    builder = SnippetPlotBuilder(title=title, scene_ranges={"xaxis": {}, "yaxis": {}, "zaxis": {}}, height=700)
    if mesh is not None:
        builder.add_mesh(mesh, double_sided=True)
    builder.fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="markers",
            marker={"size": 4, "color": "royalblue", "opacity": 0.7},
            name="Candidates",
        )
    )
    if center is not None:
        builder.fig.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="markers",
                marker={"size": 4, "color": "red", "symbol": "x"},
                name="sampling center",
            )
        )
    builder.fig.update_layout(scene={"aspectmode": "data"})
    fig = builder.finalize()
    if path is not None:
        fig.write_html(str(path))
    return fig


def plot_sampling_shell(
    shell_poses: torch.Tensor,
    last_pose: torch.Tensor | "PoseTW",  # type: ignore[name-defined]
    min_radius: float,
    max_radius: float,
    min_elev_deg: float,
    max_elev_deg: float,
    azimuth_full_circle: bool,
    mesh: trimesh.Trimesh | None,
    path: Path | None = None,
    sample_n: int = 200,
) -> go.Figure:
    """Visualise the spherical sampling band and a subset of raw shell samples."""

    center = _pose_positions(last_pose.unsqueeze(0) if isinstance(last_pose, torch.Tensor) else last_pose)[0]

    u = np.linspace(0, (2 if azimuth_full_circle else 1) * np.pi, 60)
    v = np.radians(np.linspace(min_elev_deg, max_elev_deg, 15))
    uu, vv = np.meshgrid(u, v)
    fig = go.Figure()
    # show a single mid-shell fragment with stronger alpha
    r_mid = (min_radius + max_radius) * 0.5
    xs = center[0] + r_mid * np.cos(uu) * np.cos(vv)
    ys = center[1] + r_mid * np.sin(uu) * np.cos(vv)
    zs = center[2] + r_mid * np.sin(vv)
    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=zs,
            showscale=False,
            opacity=0.6,
            surfacecolor=np.zeros_like(xs),
            name="sampling band",
            colorscale=[[0, "rgba(80,120,255,0.9)"], [1, "rgba(80,120,255,0.9)"]],
            contours={
                "x": {"show": True, "color": "#5078ff", "width": 2, "highlightwidth": 4},
                "y": {"show": True, "color": "#5078ff", "width": 2, "highlightwidth": 4},
                "z": {"show": True, "color": "#5078ff", "width": 2, "highlightwidth": 4},
            },
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode="markers",
            marker={"size": 6, "color": "black"},
            name="last pose",
        )
    )
    if mesh is not None:
        for trace in _mesh_to_plotly(mesh, double_sided=True):
            fig.add_trace(trace)
    fig.update_layout(scene_aspectmode="data", title="Sampling shell (r_min/r_max + elev cap)")
    if path is not None:
        fig.write_html(str(path))
    return fig


def plot_candidate_frustums(
    poses: torch.Tensor | "PoseTW",  # type: ignore[name-defined]
    camera,
    mesh: trimesh.Trimesh | None,
    path: Path | None = None,
    max_frustums: int = 12,
    frustum_scale: float = 1.0,
) -> go.Figure:
    """Plot mesh, candidate centres, and frustums for a subset using the shared SnippetPlotBuilder."""

    pos = _pose_positions(poses)
    builder = SnippetPlotBuilder(title="Candidates with frustums", scene_ranges={"xaxis": {}, "yaxis": {}, "zaxis": {}}, height=900)
    if mesh is not None:
        builder.add_mesh(mesh, double_sided=True)
    builder.fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="markers",
            marker={"size": 2, "color": "royalblue", "opacity": 0.35},
            name="candidates",
        )
    )

    frustum_ids = np.linspace(0, pos.shape[0] - 1, num=min(max_frustums, pos.shape[0]), dtype=int)
    cam_tw = camera.calib if hasattr(camera, "calib") else camera
    cam0 = cam_tw[0] if hasattr(cam_tw, "__getitem__") else cam_tw
    for idx in frustum_ids:
        p = poses[idx]
        if hasattr(p, "matrix"):
            t_world_cam = p.matrix.detach().cpu().numpy()
        else:
            arr = p
            if arr.dim() == 1 and arr.shape[0] == 12:
                arr = arr.view(3, 4)
            if arr.shape == torch.Size([3, 4]):
                t_world_cam = np.eye(4, dtype=np.float64)
                t_world_cam[:3, :4] = arr.detach().cpu().numpy()
            else:
                raise ValueError("Unsupported pose format for frustum plotting.")

        pose_tw = PoseTW.from_matrix(t_world_cam)
        # Use shared display transform (gravity align + display roll/yaw) for consistent axis orientation.
        pose_disp = pose_for_display(pose_tw, align_gravity=True)

        frustum_segments = _frustum_segments(cam0, pose_disp, scale=frustum_scale)
        for j, seg in enumerate(frustum_segments):
            builder.fig.add_trace(
                go.Scatter3d(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    z=seg[:, 2],
                    mode="lines",
                    line={"color": "crimson", "width": 5},
                    name="frustum" if (idx == frustum_ids[0] and j == 0) else None,
                    showlegend=bool(idx == frustum_ids[0] and j == 0),
                )
            )
        # add camera axes (RGB XYZ) for this candidate
        axes = pose_disp.rotate(torch.eye(3)).detach().cpu().numpy()
        builder.add_camera_axes(pose_disp.t.detach().cpu().numpy(), axes)

    fig = builder.finalize()
    if path is not None:
        fig.write_html(str(path))
    return fig


def main() -> None:
    # load one real snippet
    sample_cfg = AseEfmDatasetConfig(scene_ids=["81283"], verbose=False, load_meshes=True, mesh_simplify_ratio=None)
    sample = next(iter(sample_cfg.setup_target()))

    gen_cfg = CandidateViewGeneratorConfig(
        num_samples=128,
        max_resamples=2,
        ensure_collision_free=False,
        ensure_free_space=True,
        min_distance_to_mesh=0.0,
        device="cpu",
    )
    gen = CandidateViewGenerator(gen_cfg)
    result = gen.generate_from_typed_sample(sample)

    out_dir = _ensure_plot_dir()
    plot_candidates(result["poses"], sample.mesh, "Candidate positions", out_dir / "candidates.html")
    plot_sampling_shell(
        shell_poses=result["shell_poses"],
        last_pose=sample.trajectory.final_pose,
        min_radius=gen_cfg.min_radius,
        max_radius=gen_cfg.max_radius,
        min_elev_deg=gen_cfg.min_elev_deg,
        max_elev_deg=gen_cfg.max_elev_deg,
        azimuth_full_circle=gen_cfg.azimuth_full_circle,
        mesh=sample.mesh,
        path=out_dir / "sampling_shell.html",
        sample_n=300,
    )
    traj_positions = sample.trajectory.t_world_rig.matrix3x4[..., :3, 3].detach().cpu().numpy()
    plot_candidate_frustums(
        poses=result["poses"],
        camera=sample.camera_rgb,
        mesh=sample.mesh,
        traj_positions=traj_positions,
        path=out_dir / "candidate_frustums.html",
        max_frustums=6,
    )


if __name__ == "__main__":
    main()

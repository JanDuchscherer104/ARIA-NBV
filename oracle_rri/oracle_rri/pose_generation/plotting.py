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
from oracle_rri.pose_generation import CandidateViewGenerator, CandidateViewGeneratorConfig


def _ensure_plot_dir() -> Path:
    out = PathConfig().data_root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _pose_positions(poses: torch.Tensor | "PoseTW") -> np.ndarray:  # type: ignore[name-defined]
    """Extract Nx3 positions from PoseTW or (N,12)/(N,3,4)/(N,3) tensors."""
    if hasattr(poses, "matrix3x4"):
        arr = poses.matrix3x4
    elif hasattr(poses, "_data"):
        arr = poses._data  # PoseTW stores raw tensor as _data
    else:
        arr = poses
    if arr.dim() == 2 and arr.shape == torch.Size([3, 4]):
        pos = arr[..., :3, 3].unsqueeze(0)
    elif arr.dim() == 2 and arr.shape[1] == 12:
        arr = arr.view(-1, 3, 4)
        pos = arr[..., :3, 3]
    elif arr.dim() == 3 and arr.shape[1:] == (3, 4):
        pos = arr[..., :3, 3]
    elif arr.dim() == 2 and arr.shape[1] == 3:
        pos = arr
    else:
        pos = arr.t()
    return pos.detach().cpu().numpy()


def camera_intrinsics(cam, frame_idx: int = 0) -> np.ndarray:
    """Return 3x3 pinhole intrinsics; accepts EfmCameraView or CameraTW."""
    cam_tw = cam.calib if hasattr(cam, "calib") else cam
    data = cam_tw.tensor()
    fx, fy = data[frame_idx, cam_tw.F_IND].tolist()
    cx, cy = data[frame_idx, cam_tw.C_IND].tolist()
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


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
) -> go.Figure:
    """Plot candidate camera centres with optional mesh overlay.

    Args:
        poses: Candidate poses as PoseTW or tensor ``[N,12]/[N,3,4]``.
        mesh: Optional scene mesh to show as translucent surface.
        title: Plot title.
        path: Optional output path; if provided, the figure is written as HTML.

    Returns:
        Plotly figure ready for inline rendering (e.g. Streamlit).
    """
    fig = go.Figure()
    pos = _pose_positions(poses)
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="markers",
            marker={"size": 3, "color": "blue", "opacity": 0.6},
            name="candidates",
        )
    )
    if mesh is not None:
        verts = mesh.vertices
        faces = mesh.faces
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.15,
                color="gray",
                name="mesh",
            )
        )
    fig.update_layout(scene_aspectmode="data", title=title)
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

    pos = _pose_positions(shell_poses)
    center = _pose_positions(last_pose.unsqueeze(0) if isinstance(last_pose, torch.Tensor) else last_pose)[0]

    # Candidate subset for clarity
    if pos.shape[0] > sample_n:
        idx = np.random.choice(pos.shape[0], sample_n, replace=False)
        pos_sub = pos[idx]
    else:
        pos_sub = pos

    u = np.linspace(0, (2 if azimuth_full_circle else 1) * np.pi, 60)
    v = np.radians(np.linspace(min_elev_deg, max_elev_deg, 30))
    uu, vv = np.meshgrid(u, v)
    fig = go.Figure()
    sphere_specs = [
        (min_radius, "rgba(255,80,80,0.35)", "min radius"),
        (max_radius, "rgba(80,120,255,0.25)", "max radius"),
    ]
    for r, color, name in sphere_specs:
        xs = center[0] + r * np.cos(uu) * np.cos(vv)
        ys = center[1] + r * np.sin(uu) * np.cos(vv)
        zs = center[2] + r * np.sin(vv)
        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                showscale=False,
                opacity=0.35,
                surfacecolor=np.zeros_like(xs),
                name=name,
                colorscale=[[0, color], [1, color]],
                contours={
                    "x": {"show": True, "color": color, "width": 2, "highlightwidth": 4},
                    "y": {"show": True, "color": color, "width": 2, "highlightwidth": 4},
                    "z": {"show": True, "color": color, "width": 2, "highlightwidth": 4},
                },
            )
        )
    # add an equator ring to make the band visible
    ring_theta = np.linspace(min_elev_deg, max_elev_deg, 3)
    for theta_deg, ring_color in zip(ring_theta, ["#ff5050", "#8080ff", "#5050ff"], strict=False):
        theta = np.radians(theta_deg)
        ring_r = (min_radius + max_radius) * 0.5
        x_ring = center[0] + ring_r * np.cos(u) * np.cos(theta)
        y_ring = center[1] + ring_r * np.sin(u) * np.cos(theta)
        z_ring = center[2] + ring_r * np.sin(theta) * np.ones_like(u)
        fig.add_trace(
            go.Scatter3d(
            x=x_ring,
            y=y_ring,
            z=z_ring,
            mode="lines",
            line={"color": ring_color, "width": 6, "dash": "dash"},
            name=f"elev {theta_deg:.1f}°",
            showlegend=False,
            opacity=0.8,
        )
    )

        fig.add_trace(
            go.Scatter3d(
                x=pos_sub[:, 0],
                y=pos_sub[:, 1],
                z=pos_sub[:, 2],
                mode="markers",
                marker={"size": 3, "color": "darkorange", "opacity": 0.6},
                name="shell samples",
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
        verts = mesh.vertices
        faces = mesh.faces
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.15,
                color="gray",
                name="mesh",
                hoverinfo="skip",
            )
        )
    fig.update_layout(scene_aspectmode="data", title="Sampling shell (r_min/r_max + elev cap)")
    if path is not None:
        fig.write_html(str(path))
    return fig


def plot_candidate_frustums(
    poses: torch.Tensor | "PoseTW",  # type: ignore[name-defined]
    camera,
    mesh: trimesh.Trimesh | None,
    traj_positions: np.ndarray,
    path: Path | None = None,
    max_frustums: int = 12,
) -> go.Figure:
    """Plot trajectory, mesh, all candidate centres, and frustums for a few."""

    pos = _pose_positions(poses)
    fig = go.Figure()

    if mesh is not None:
        verts = mesh.vertices
        faces = mesh.faces
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.18,
                color="lightgray",
                name="mesh",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=traj_positions[:, 0],
            y=traj_positions[:, 1],
            z=traj_positions[:, 2],
            mode="lines+markers",
            marker={"size": 3, "color": "black"},
            line={"color": "black", "width": 4},
            name="trajectory",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="markers",
            marker={"size": 2, "color": "royalblue", "opacity": 0.4},
            name="candidates",
        )
    )

    intrinsics = camera_intrinsics(camera, frame_idx=0)
    cam_tw = camera.calib if hasattr(camera, "calib") else camera
    cam_data = cam_tw.tensor()
    if hasattr(camera, "images"):
        image_h = int(camera.images.shape[-2])
        image_w = int(camera.images.shape[-1])
    else:
        image_h = int(cam_data[0, cam_tw.SIZE_IND][1].item()) if hasattr(cam_tw, "SIZE_IND") else 480
        image_w = int(cam_data[0, cam_tw.SIZE_IND][0].item()) if hasattr(cam_tw, "SIZE_IND") else 640

    frustum_ids = np.linspace(0, pos.shape[0] - 1, num=min(max_frustums, pos.shape[0]), dtype=int)
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
        segments = camera_wireframe_segments(intrinsics, t_world_cam, (image_w, image_h))
        for j, seg in enumerate(segments):
            fig.add_trace(
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

    fig.update_layout(title="Candidates with frustums + trajectory", scene={"aspectmode": "data"}, height=900)
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

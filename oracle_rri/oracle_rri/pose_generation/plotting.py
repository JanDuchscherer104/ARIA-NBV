"""Plotting helpers for candidate sampling."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go  # type: ignore[import]
import torch
from efm3d.aria.pose import PoseTW
from plotly.subplots import make_subplots  # type: ignore[import]

from oracle_rri.data import AseEfmDatasetConfig, EfmCameraView, EfmSnippetView
from oracle_rri.data.plotting import SnippetPlotBuilder
from oracle_rri.pose_generation import CandidateViewGenerator, CandidateViewGeneratorConfig
from oracle_rri.utils import Console, Verbosity

console = Console.with_prefix("pose_plotting")


def plot_candidate_frusta(
    *,
    snippet: EfmSnippetView,
    poses: PoseTW,
    camera_view: EfmCameraView,
    frustum_scale: float,
    max_frustums: int,
) -> go.Figure:
    """Plot mesh, candidate centres, and frustums using the shared builder."""

    calib = camera_view.calib
    num_frames = calib.shape[0] if calib.ndim > 1 else 1
    cams = [calib[idx] for idx in range(num_frames)] if num_frames > 1 else [calib]

    return (
        SnippetPlotBuilder.from_snippet(snippet, title="Candidates with frustums")
        .add_mesh()
        .add_points(poses, name="Candidates", color="royalblue", size=3, opacity=0.35)
        .add_candidate_frusta(
            cam=cams,
            poses=poses,
            scale=frustum_scale,
            color="crimson",
            name="Frustum",
            max_frustums=max_frustums,
            include_axes=False,
            include_center=False,
        )
    ).finalize()


def plot_candidates(
    *,
    snippet: EfmSnippetView,
    poses: PoseTW,
    title: str = "Candidate positions",
    center: np.ndarray | None = None,
) -> go.Figure:
    """Plot candidate camera centres with optional mesh overlay."""
    builder = (
        SnippetPlotBuilder.from_snippet(snippet, title=title)
        .add_mesh()
        .add_points(poses, name="Candidates", color="royalblue", size=4, opacity=0.7)
        .add_camera_axes(camera="rgb", title="Final Camera")
    )

    if center is not None:
        builder = builder.add_points(
            np.asarray(center).reshape(1, 3),
            name="sampling center",
            color="red",
            size=1,
            symbol="x",
            opacity=1.0,
        )

    return builder.finalize()


def plot_direction_polar(
    dirs: torch.Tensor,
    *,
    title: str = "Direction distribution (az/elev)",
    bins: int = 40,
) -> go.Figure:
    """Plot azimuth/elevation density of direction vectors."""

    d = dirs.detach().cpu().numpy()
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    elev = np.arcsin(d[:, 1])  # y is up in LUF
    az = np.arctan2(d[:, 0], d[:, 2])  # atan2(x, z) per our sampling
    elev_deg = np.degrees(elev)
    az_deg = np.degrees(az)
    h, xedges, yedges = np.histogram2d(az_deg, elev_deg, bins=bins)
    fig = go.Figure(
        data=go.Heatmap(
            x=xedges[:-1],
            y=yedges[:-1],
            z=h.T,
            colorscale="Viridis",
            colorbar_title="count",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Azimuth (deg)",
        yaxis_title="Elevation (deg)",
        yaxis={"scaleanchor": None},
    )
    return fig


def plot_direction_sphere(
    dirs: torch.Tensor,
    *,
    title: str = "Directions on unit sphere",
    sample_n: int = 2000,
    show_axes: bool = False,
) -> go.Figure:
    """3D scatter of directions on the unit sphere."""

    d = dirs.detach().cpu()
    if d.shape[0] > sample_n:
        idx = torch.linspace(0, d.shape[0] - 1, steps=sample_n, dtype=torch.long)
        d = d[idx]
    d = d / (d.norm(dim=1, keepdim=True) + 1e-8)
    dn = d.numpy()
    traces = [
        go.Scatter3d(
            x=dn[:, 0],
            y=dn[:, 1],
            z=dn[:, 2],
            mode="markers",
            marker={"size": 2, "color": dn[:, 1], "colorscale": "Turbo", "opacity": 0.7},
            name="dirs",
        )
    ]

    if show_axes:
        axes = np.eye(3)
        labels = ["X (left)", "Y (up)", "Z (fwd)"]
        colors = ["red", "green", "blue"]
        for i in range(3):
            traces.append(
                go.Scatter3d(
                    x=[0, axes[i, 0]],
                    y=[0, axes[i, 1]],
                    z=[0, axes[i, 2]],
                    mode="lines+markers",
                    line={"color": colors[i], "width": 6},
                    marker={"size": 3, "color": colors[i]},
                    name=labels[i],
                    showlegend=True,
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(title=title, scene={"xaxis_title": "X (left)", "yaxis_title": "Y (up)", "zaxis_title": "Z (fwd)"})
    return fig


def plot_position_polar(
    offsets: torch.Tensor,
    *,
    title: str = "Position distribution (az/elev)",
    bins: int = 40,
) -> go.Figure:
    """Polar heatmap of offsets (converted to directions)."""

    # Convert to unit directions
    dirs = offsets / (offsets.norm(dim=1, keepdim=True) + 1e-8)
    return plot_direction_polar(dirs, title=title, bins=bins)


def plot_position_sphere(
    offsets: torch.Tensor,
    *,
    title: str = "Positions on unit sphere",
    sample_n: int = 2000,
    show_axes: bool = False,
) -> go.Figure:
    """3D scatter of position offsets projected to the unit sphere."""

    dirs = offsets / (offsets.norm(dim=1, keepdim=True) + 1e-8)
    return plot_direction_sphere(dirs, title=title, sample_n=sample_n, show_axes=show_axes)


def plot_direction_marginals(dirs: torch.Tensor, bins: int = 60) -> go.Figure:
    d = dirs.detach().cpu().numpy()
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    elev = np.arcsin(d[:, 1])
    az = np.arctan2(d[:, 0], d[:, 2])

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Azimuth", "Elevation"))
    fig.add_histogram(x=np.degrees(az), nbinsx=bins, row=1, col=1)
    fig.add_histogram(x=np.degrees(elev), nbinsx=bins, row=1, col=2)
    fig.update_xaxes(title="deg", row=1, col=1)
    fig.update_xaxes(title="deg", row=1, col=2)
    return fig


def plot_rule_masks(
    *,
    snippet: EfmSnippetView,
    shell_poses: torch.Tensor,
    masks: torch.Tensor,
    rule_names: list[str],
    sample_n: int = 500,
) -> go.Figure:
    """Visualise per-rule pruning on raw samples."""

    shell_np = shell_poses.detach().cpu().numpy()
    pts_all = shell_np[:, 9:12] if shell_np.shape[1] == 12 else shell_np
    # Align lengths defensively in case masks and shell_poses differ (e.g., legacy caches).
    max_len = min(sample_n, pts_all.shape[0], masks.shape[1])
    pts = pts_all[:max_len]
    builder = SnippetPlotBuilder.from_snippet(snippet, title="Rule-wise pruning").add_mesh()
    palette = ["#4caf50", "#f44336", "#2196f3", "#ff9800", "#9c27b0"]
    for ridx, name in enumerate(rule_names):
        if masks.shape[0] <= ridx:
            break
        valid = masks[ridx, :max_len].detach().cpu().numpy().astype(bool)
        color = palette[ridx % len(palette)]
        builder = builder.add_points(
            pts[valid],
            name=f"{name}: kept",
            color=color,
            size=3,
            opacity=0.65,
        )
    return builder.finalize()


def plot_candidate_axes(
    *,
    snippet: EfmSnippetView,
    poses: PoseTW,
    scale: float = 0.25,
    max_axes: int = 50,
) -> go.Figure:
    """Plot candidate centres with their local axes to inspect roll/yaw."""

    cam = snippet.camera_rgb.calib[0]
    builder = (
        SnippetPlotBuilder.from_snippet(snippet, title="Candidate axes")
        .add_mesh()
        .add_candidate_frusta(
            cam=[cam],
            poses=poses[:max_axes],
            scale=scale,
            color="crimson",
            name="candidates",
            include_axes=True,
            include_center=True,
            max_frustums=max_axes,
        )
    )
    return builder.finalize()


def main() -> None:
    # load one real snippet
    sample_cfg = AseEfmDatasetConfig(
        scene_ids=["81283"],
        verbosity=Verbosity.QUIET,
        load_meshes=True,
        mesh_simplify_ratio=None,
    )
    sample = next(iter(sample_cfg.setup_target()))

    gen_cfg = CandidateViewGeneratorConfig(
        num_samples=128,
        max_resamples=2,
        ensure_collision_free=False,
        ensure_free_space=True,
        min_distance_to_mesh=0.0,
        device=torch.device("cpu"),
    )
    gen = CandidateViewGenerator(gen_cfg)
    result = gen.generate_from_typed_sample(sample)

    plot_candidates(snippet=sample, poses=result["poses"], title="Candidate positions")
    plot_sampling_shell(
        snippet=sample,
        shell_poses=result["shell_poses"],
        last_pose=sample.trajectory.final_pose,
        min_radius=gen_cfg.min_radius,
        max_radius=gen_cfg.max_radius,
        min_elev_deg=gen_cfg.min_elev_deg,
        max_elev_deg=gen_cfg.max_elev_deg,
        azimuth_full_circle=gen_cfg.azimuth_full_circle,
        sample_n=300,
    )


if __name__ == "__main__":
    main()

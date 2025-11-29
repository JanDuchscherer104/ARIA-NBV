"""Plotting helpers for candidate sampling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import plotly.graph_objects as go  # type: ignore[import]
import torch
from efm3d.aria.pose import PoseTW
from plotly.subplots import make_subplots  # type: ignore[import]

from oracle_rri.data import EfmSnippetView
from oracle_rri.data.plotting import SnippetPlotBuilder
from oracle_rri.utils import Console

from .types import CandidateSamplingResult

console = Console.with_prefix("pose_plotting")


class CandidatePlotBuilder(SnippetPlotBuilder):
    candidate_results: CandidateSamplingResult | None = None
    candidate_cfg = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_results = None

    def attach_candidate_results(self, results: CandidateSamplingResult) -> CandidatePlotBuilder:
        """Attach candidate sampling results for plotting."""
        self.candidate_results = results
        return self

    def attach_candidate_cfg(self, cfg) -> CandidatePlotBuilder:
        """Attach candidate config for metadata-aware plotting."""
        self.candidate_cfg = cfg
        return self

    # ---------------- world-frame helpers ----------------
    def _world_positions(self, use_valid: bool = True) -> np.ndarray:
        if self.candidate_results is None:
            raise ValueError("Candidate results missing; call attach_candidate_results() first.")
        shell = self.candidate_results.shell_poses
        mask = self.candidate_results.mask_valid if use_valid else None
        if mask is not None:
            shell = PoseTW(shell._data[mask])
        return shell.t.detach().cpu().numpy()

    def add_candidate_points(
        self,
        *,
        use_valid: bool = True,
        color: np.ndarray | None = None,
        colorbar_title: str | None = None,
        name: str = "Candidates",
        size: int = 3,
        opacity: float = 0.7,
        hovertext: list[str] | None = None,
    ) -> "CandidatePlotBuilder":
        pts = self._world_positions(use_valid=use_valid)
        marker = {"size": size, "opacity": opacity}
        if color is not None:
            marker.update({"color": color, "colorscale": "Viridis"})
            if colorbar_title:
                marker["colorbar"] = {"title": colorbar_title}
        self.fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=marker,
                name=name,
                hovertext=hovertext,
                hoverinfo="text" if hovertext else "name",
            )
        )
        return self

    def add_min_distance_overlay(self, distances: torch.Tensor, *, use_valid: bool = False) -> "CandidatePlotBuilder":
        dist_np = distances.detach().cpu().numpy().reshape(-1)
        mask = self.candidate_results.mask_valid.detach().cpu().numpy()
        hover = [f"dist={d:.3f} m<br>valid={bool(v)}" for d, v in zip(dist_np.tolist(), mask.tolist(), strict=False)]
        return self.add_candidate_points(
            use_valid=use_valid,
            color=dist_np,
            colorbar_title="Min dist (m)",
            name="Candidates",
            hovertext=hover,
            opacity=0.8,
            size=4,
        )

    def add_path_collision_segments(self, collision_mask: torch.Tensor) -> "CandidatePlotBuilder":
        ref = self.candidate_results.reference_pose.t.detach().cpu().numpy()
        centers = self.candidate_results.shell_poses.t.detach().cpu().numpy()
        mask_np = collision_mask.detach().cpu().numpy().astype(bool)
        rej_centers = centers[mask_np]
        if rej_centers.size > 0:
            seg_pts = []
            for c in rej_centers:
                seg_pts.append(ref)
                seg_pts.append(c)
                seg_pts.append([np.nan, np.nan, np.nan])
            seg_pts = np.array(seg_pts)
            self.fig.add_trace(
                go.Scatter3d(
                    x=seg_pts[:, 0],
                    y=seg_pts[:, 1],
                    z=seg_pts[:, 2],
                    mode="lines",
                    line={"color": "crimson", "width": 4},
                    name="Rejected path",
                    hoverinfo="skip",
                )
            )
        self.fig.add_trace(
            go.Scatter3d(
                x=[ref[0]],
                y=[ref[1]],
                z=[ref[2]],
                mode="markers",
                marker={"color": "black", "size": 6, "symbol": "diamond"},
                name="Reference pose",
            )
        )
        return self

    def rule_rejection_bar(self) -> go.Figure:
        masks = self.candidate_results.masks if self.candidate_results is not None else {}
        if not isinstance(masks, dict) or len(masks) == 0:
            fig = go.Figure()
            fig.update_layout(title="Rule rejections (no masks collected)")
            return fig
        prev = torch.ones_like(next(iter(masks.values())), dtype=torch.bool)
        names: list[str] = []
        counts: list[int] = []
        for name, mask in masks.items():
            rej = prev & (~mask)
            names.append(name)
            counts.append(int(rej.sum().item()))
            prev = mask
        fig = go.Figure(go.Bar(x=names, y=counts, marker_color="steelblue"))
        fig.update_layout(title="Rejections per rule", xaxis_title="Rule", yaxis_title="# rejected")
        return fig

    def add_candidate_frusta(
        self,
        *,
        scale: float = 1.0,
        color: str = "crimson",
        name: str = "Frustum",
        max_frustums: int | None = None,
        include_axes: bool = False,
        include_center: bool = False,
    ) -> "SnippetPlotBuilder":
        """Overlay frusta using the attached candidate results."""

        cand_results = self.candidate_results
        if cand_results is None:
            raise ValueError("Candidate results missing; call attach_candidate_results() first.")

        cams = cand_results.views
        poses_world_valid = PoseTW(cand_results.shell_poses._data[cand_results.mask_valid])
        pose_list = self._pose_list_from_input(poses_world_valid)

        return self._add_frusta_for_poses(
            cams=cams,
            poses=pose_list,
            scale=scale,
            color=color,
            name=name,
            max_frustums=max_frustums,
            include_axes=include_axes,
            include_center=include_center,
        )


def plot_candidate_frusta(
    *,
    snippet: EfmSnippetView,
    candidates: CandidateSamplingResult,
    frustum_scale: float,
    max_frustums: int,
) -> go.Figure:
    """Plot mesh, candidate centres, and frustums using the shared builder."""

    return (
        CandidatePlotBuilder.from_snippet(snippet, title="Candidates with frustums")
        .attach_candidate_results(candidates)
        .add_mesh()
        .add_candidate_points(use_valid=True, name="Candidates", opacity=0.35, size=3, color="royalblue")
        .add_candidate_frusta(
            scale=frustum_scale,
            color="crimson",
            name="Frustum",
            max_frustums=max_frustums,
            include_axes=False,
            include_center=False,
        )
        .add_frame_axes(frame="rgb", title="Final Camera")
    ).finalize()


def plot_candidates(
    *,
    snippet: EfmSnippetView,
    poses: PoseTW,
    title: str = "Candidate positions",
    center: np.ndarray | None = None,
    camera: Literal["rgb", "slaml", "slamr"] = "rgb",
    camera_frame_indices: Sequence[int] | None = None,
) -> go.Figure:
    """Plot candidate camera centres with optional mesh overlay."""
    builder = (
        SnippetPlotBuilder.from_snippet(snippet, title=title)
        .add_mesh()
        .add_points(poses, name="Candidates", color="royalblue", size=4, opacity=0.7)
    )

    if camera_frame_indices is not None:
        builder = builder.add_frame_axes(frame=camera, frame_indices=list(camera_frame_indices), title="Final Camera")
    else:
        builder = builder.add_frame_axes(frame=camera, title="Final Camera", frame_indices=camera_frame_indices)

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


def candidate_offsets_and_dirs_ref(
    candidates: CandidateSamplingResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offsets and forward directions in reference frame for **valid** candidates."""

    poses_ref = candidates.views.T_camera_rig  # reference←cam for valid mask
    offsets = poses_ref.t
    offsets = offsets.view(-1, 3)
    z_cam = (
        torch.tensor([0.0, 0.0, 1.0], device=offsets.device, dtype=offsets.dtype).view(1, 3).expand(offsets.shape[0], 3)
    )
    dirs = poses_ref.rotate(z_cam).view(-1, 3)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
    return offsets, dirs


def shell_offsets_and_dirs_ref(
    candidates: CandidateSamplingResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offsets and forward directions in reference frame for the full shell (pre-pruning)."""

    ref = candidates.reference_pose
    ref_inv = ref.inverse()
    shell = candidates.shell_poses
    centers_world = shell.t
    offsets_ref = ref_inv.transform(centers_world).view(-1, 3)

    z_cam = (
        torch.tensor([0.0, 0.0, 1.0], device=shell.R.device, dtype=shell.R.dtype).view(1, 3).expand(shell.R.shape[0], 3)
    )
    dirs_world = shell.rotate(z_cam)
    dirs_ref = ref_inv.rotate(dirs_world).view(-1, 3)
    dirs_ref = dirs_ref / (dirs_ref.norm(dim=1, keepdim=True) + 1e-8)
    return offsets_ref, dirs_ref


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
    title: str = "Position distribution (radius vs elev)",
    bins: int = 40,
) -> go.Figure:
    """Heatmap of radius vs elevation for position offsets."""

    off = offsets.detach().cpu().numpy()
    r = np.linalg.norm(off, axis=1)
    elev = np.degrees(np.arcsin(off[:, 1] / (r + 1e-8)))
    h, xedges, yedges = np.histogram2d(r, elev, bins=bins)
    fig = go.Figure(
        data=go.Heatmap(
            x=xedges[:-1],
            y=yedges[:-1],
            z=h.T,
            colorscale="Viridis",
            colorbar_title="count",
        )
    )
    fig.update_layout(title=title, xaxis_title="radius (m)", yaxis_title="elev (deg)")
    return fig


def plot_position_sphere(
    offsets: torch.Tensor,
    *,
    title: str = "Positions in rig frame",
    sample_n: int = 2000,
    show_axes: bool = False,
) -> go.Figure:
    """3D scatter of position offsets (not normalised)."""

    pts = offsets.detach().cpu()
    if pts.shape[0] > sample_n:
        idx = torch.linspace(0, pts.shape[0] - 1, steps=sample_n, dtype=torch.long)
        pts = pts[idx]
    pn = pts.numpy()
    traces = [
        go.Scatter3d(
            x=pn[:, 0],
            y=pn[:, 1],
            z=pn[:, 2],
            mode="markers",
            marker={"size": 2, "color": pn[:, 1], "colorscale": "Turbo", "opacity": 0.7},
            name="positions",
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
    fig.update_layout(
        title=title,
        scene={"xaxis_title": "X (left)", "yaxis_title": "Y (up)", "zaxis_title": "Z (fwd)"},
    )
    return fig


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


def plot_radius_hist(
    offsets: torch.Tensor,
    *,
    title: str = "Radius distribution",
    bins: int = 40,
) -> go.Figure:
    r = offsets.norm(dim=1).detach().cpu().numpy()
    fig = go.Figure(go.Histogram(x=r, nbinsx=bins))
    fig.update_layout(
        title=title,
        xaxis_title="radius (m)",
        yaxis_title="count",
    )
    return fig


def plot_min_distance_to_mesh(
    *,
    snippet: EfmSnippetView,
    candidates: CandidateSamplingResult,
    distances: torch.Tensor,
) -> go.Figure:
    """Binary-coloured candidates: rejected (red, opaque) vs accepted (green, faint)."""

    dist_np = distances.detach().cpu().numpy().reshape(-1)
    mask_valid = candidates.mask_valid.detach().cpu().numpy().astype(bool)
    colors = ["rgba(0,200,0,0.2)" if v else "rgba(220,0,0,0.9)" for v in mask_valid.tolist()]
    hover = [f"dist={d:.3f} m<br>valid={bool(v)}" for d, v in zip(dist_np.tolist(), mask_valid.tolist(), strict=False)]

    builder = (
        CandidatePlotBuilder.from_snippet(snippet, title="Min distance to mesh")
        .attach_candidate_results(candidates)
        .add_mesh()
        .add_candidate_points(
            use_valid=False,
            color=np.array(colors),
            name="Candidates",
            opacity=1.0,
            size=4,
            hovertext=hover,
        )
        .add_frame_axes(frame="rig", title="Reference frame")
    )
    return builder.finalize()


def plot_path_collision_segments(
    *,
    snippet: EfmSnippetView,
    candidates: CandidateSamplingResult,
    collision_mask: torch.Tensor,
) -> go.Figure:
    """Plot segments from reference pose to candidates rejected by path-collision rule."""

    mask_np = collision_mask.detach().cpu().numpy().astype(bool)
    colors = ["rgba(0,200,0,0.2)" if not m else "rgba(220,0,0,0.9)" for m in mask_np.tolist()]

    builder = (
        CandidatePlotBuilder.from_snippet(snippet, title="Path collision segments")
        .attach_candidate_results(candidates)
        .add_mesh()
    )

    if mask_np.any():
        builder.add_path_collision_segments(collision_mask)

    builder.add_candidate_points(use_valid=False, color=np.array(colors), name="Candidates", opacity=1.0, size=4)
    builder.add_frame_axes(frame="rig", title="Reference frame")
    return builder.finalize()
def plot_rule_rejection_bar(candidates: CandidateSamplingResult) -> go.Figure:
    """Bar chart of rejection counts per rule using cumulative masks."""

    masks = candidates.masks
    if not isinstance(masks, dict) or len(masks) == 0:
        fig = go.Figure()
        fig.update_layout(title="Rule rejections (no masks collected)")
        return fig

    prev = torch.ones_like(next(iter(masks.values())), dtype=torch.bool)
    names: list[str] = []
    counts: list[int] = []
    for name, mask in masks.items():
        rej = prev & (~mask)
        names.append(name)
        counts.append(int(rej.sum().item()))
        prev = mask

    fig = go.Figure(go.Bar(x=names, y=counts, marker_color="steelblue"))
    fig.update_layout(title="Rejections per rule", xaxis_title="Rule", yaxis_title="# rejected")
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

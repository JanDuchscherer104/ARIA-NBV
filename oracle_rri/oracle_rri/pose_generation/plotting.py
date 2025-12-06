"""Plotting helpers for candidate sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
import plotly.graph_objects as go  # type: ignore[import]
import torch
from efm3d.aria.pose import PoseTW
from plotly.subplots import make_subplots  # type: ignore[import]

from oracle_rri.data import EfmSnippetView
from oracle_rri.data.plotting import SnippetPlotBuilder, rotate_yaw_cw90
from oracle_rri.utils import Console

if TYPE_CHECKING:
    from .candidate_generation import CandidateViewGeneratorConfig
    from .types import CandidateSamplingResult

console = Console.with_prefix("pose_plotting")


class CandidatePlotBuilder(SnippetPlotBuilder):
    candidate_results: CandidateSamplingResult | None = None
    candidate_cfg: CandidateViewGeneratorConfig | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_results = None
        self.candidate_cfg = None
        self._centers_valid: np.ndarray | None = None
        self._centers_all: np.ndarray | None = None
        self._ref_center: np.ndarray | None = None

    @classmethod
    def from_candidates(
        cls, snippet: EfmSnippetView, candidates: CandidateSamplingResult, *, title: str, height: int = 900
    ) -> Self:
        return cls.from_snippet(snippet, title=title, height=height).attach_candidate_results(candidates)

    def attach_candidate_results(self, results: CandidateSamplingResult) -> Self:
        """Attach candidate sampling results for plotting."""
        self.candidate_results = results
        self._centers_all = None
        self._centers_valid = None
        self._ref_center = None
        return self

    def attach_candidate_cfg(self, cfg: CandidateViewGeneratorConfig) -> Self:
        """Attach candidate config for metadata-aware plotting."""
        self.candidate_cfg = cfg
        return self

    # ---------------- world-frame helpers ----------------
    def _world_positions(self, use_valid: bool = True) -> np.ndarray:
        if self.candidate_results is None:
            raise ValueError("Candidate results missing; call attach_candidate_results() first.")
        if use_valid:
            if self._centers_valid is None:
                shell = PoseTW(self.candidate_results.shell_poses._data[self.candidate_results.mask_valid])
                self._centers_valid = shell.t.detach().cpu().numpy()
            return self._centers_valid
        if self._centers_all is None:
            self._centers_all = self.candidate_results.shell_poses.t.detach().cpu().numpy()
        return self._centers_all

    def _mask_valid_np(self) -> np.ndarray:
        if self.candidate_results is None:
            raise ValueError("Candidate results missing; call attach_candidate_results() first.")
        return self.candidate_results.mask_valid.detach().cpu().numpy()

    def _ref_center_np(self) -> np.ndarray:
        if self.candidate_results is None:
            raise ValueError("Candidate results missing; call attach_candidate_results() first.")
        if self._ref_center is None:
            self._ref_center = self.candidate_results.reference_pose.t.detach().cpu().numpy()
        return self._ref_center

    def add_reference_axes(self, *, title: str = "Reference frame") -> Self:
        if self.candidate_results is None:
            return self
        return self.add_frame_axes(frame=self.candidate_results.reference_pose, title=title)

    def add_candidate_points(
        self,
        *,
        use_valid: bool = True,
        color: np.ndarray | str | None = None,
        colorbar_title: str | None = None,
        name: str = "Candidates",
        size: int = 3,
        opacity: float = 0.7,
        hovertext: list[str] | None = None,
        mark_reference: bool = False,
        reference_symbol: str = "diamond",
    ) -> Self:
        pts = self._world_positions(use_valid=use_valid)
        marker = {"size": size, "opacity": opacity}
        if color is not None:
            if isinstance(color, np.ndarray):
                marker.update({"color": color, "colorscale": "Viridis"})
                if colorbar_title:
                    marker["colorbar"] = {"title": colorbar_title}
            else:
                marker.update({"color": color})
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
        if mark_reference and self.candidate_results is not None:
            ref = self.candidate_results.reference_pose.t.detach().cpu().numpy()
            self.fig.add_trace(
                go.Scatter3d(
                    x=[ref[0]],
                    y=[ref[1]],
                    z=[ref[2]],
                    mode="markers",
                    marker={"color": "black", "size": 6, "symbol": reference_symbol},
                    name="Reference pose",
                )
            )
        return self

    def add_candidate_cloud(
        self,
        *,
        use_valid: bool = True,
        color: str | np.ndarray | None = "royalblue",
        name: str = "Candidates",
        size: int = 4,
        opacity: float = 0.7,
        mark_reference: bool = True,
    ) -> Self:
        return self.add_candidate_points(
            use_valid=use_valid,
            color=color,
            name=name,
            size=size,
            opacity=opacity,
            mark_reference=mark_reference,
        )

    def add_rejected_cloud(
        self,
        *,
        color: str = "crimson",
        name: str = "Rejected",
        size: int = 4,
        opacity: float = 0.8,
    ) -> Self:
        if self.candidate_results is None:
            return self
        mask = self._mask_valid_np()
        if mask.size == 0 or mask.all():
            return self
        pts = self._world_positions(use_valid=False)[~mask]
        return self.add_points(pts, name=name, color=color, size=size, opacity=opacity)

    def add_min_distance_overlay(self, distances: torch.Tensor, *, use_valid: bool = False) -> Self:
        dist_np = distances.detach().cpu().numpy().reshape(-1)
        mask = self._mask_valid_np()
        hover = [f"dist={d:.3f} m<br>valid={bool(v)}" for d, v in zip(dist_np.tolist(), mask.tolist(), strict=False)]
        return self.add_candidate_points(
            use_valid=use_valid,
            color=dist_np,
            colorbar_title="Min dist (m)",
            name="Candidates",
            hovertext=hover,
            opacity=0.8,
            size=4,
            mark_reference=True,
        )

    def add_path_collision_segments(self, collision_mask: torch.Tensor) -> Self:
        ref = self._ref_center_np()
        centers = self._world_positions(use_valid=False)
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
        assert cand_results.shell_poses._data is not None
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


def candidate_offsets_and_dirs_ref(
    candidates: CandidateSamplingResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offsets and forward directions in reference frame for **valid** candidates."""

    poses_ref = rotate_yaw_cw90(
        candidates.views.T_camera_rig
    )  # reference2candidate_cam, account for cw90 of reference frame
    offsets = poses_ref.inverse().t  # camera2reference -> offset in reference frame
    offsets = offsets.view(-1, 3)
    z_cam = (
        torch.tensor([0.0, 0.0, 1.0], device=offsets.device, dtype=offsets.dtype).view(1, 3).expand(offsets.shape[0], 3)
    )
    # Forward in reference frame: transform camera +z into reference using reference←camera.
    poses_ref_refcam = poses_ref.inverse()
    dirs = poses_ref_refcam.rotate(z_cam).view(-1, 3)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
    return offsets, dirs


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
    fig = go.Figure(data=traces)

    if show_axes:
        fig = SnippetPlotBuilder.add_frame_axes_to_fig(
            fig=fig, cam_centers=np.zeros((1, 3)), cam_axes=np.eye(3), scale=0.4
        )

    fig.update_layout(title=title, scene={"xaxis_title": "X (left)", "yaxis_title": "Y (up)", "zaxis_title": "Z (fwd)"})
    return fig


def plot_position_polar(
    offsets: torch.Tensor, *, title: str = "Offsets from reference pose (az/elev)", bins: int = 72
) -> go.Figure:
    off = offsets.detach().cpu().numpy()
    # LUF: x=left, y=up, z=forward
    az = np.degrees(np.arctan2(off[:, 0], off[:, 2]))  # atan2(x, z)
    el = np.degrees(np.arctan2(off[:, 1], np.linalg.norm(off[:, [0, 2]], axis=1) + 1e-8))
    h, xedges, yedges = np.histogram2d(az, el, bins=bins)
    fig = go.Figure(go.Heatmap(x=xedges[:-1], y=yedges[:-1], z=h.T, colorscale="Viridis", colorbar_title="count"))
    fig.update_layout(title=title, xaxis_title="azimuth (deg)", yaxis_title="elevation (deg)")
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

    fig = go.Figure(
        data=go.Scatter3d(
            x=pn[:, 0],
            y=pn[:, 1],
            z=pn[:, 2],
            mode="markers",
            marker={"size": 2, "color": pn[:, 1], "colorscale": "Turbo", "opacity": 0.7},
            name="positions",
        )
    )
    if show_axes:
        fig = SnippetPlotBuilder.add_frame_axes_to_fig(
            fig=fig, cam_centers=np.zeros((1, 3)), cam_axes=np.eye(3), scale=0.4
        )
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
        .add_reference_axes()
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
    builder.add_reference_axes()
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

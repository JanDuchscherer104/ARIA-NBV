"""Plotting utilities for VIN encodings and pose descriptors (Plotly)."""

from __future__ import annotations

import math
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.io as pio
import seaborn as sns
import torch
from e3nn import o3  # type: ignore[import-untyped]
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from plotly import colors as plotly_colors  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
from pytorch3d.renderer.cameras import (
    PerspectiveCameras,  # type: ignore[import-untyped]
)

from oracle_rri.utils.frames import rotate_yaw_cw90

from ..data.efm_views import EfmSnippetView
from ..data.plotting import SnippetPlotBuilder
from .model import _build_frustum_points_world_p3d
from .types import VinForwardDiagnostics

if TYPE_CHECKING:
    from .pose_encoding import LearnableFourierFeatures


@dataclass
class PlottingConfig:
    """Reusable plotting style that can be applied across figures."""

    style: str = "whitegrid"
    palette: str | list[str] = "tab10"
    font_family: str = "DejaVu Sans"
    font_scale: float = 1.0
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    figure_dpi: int = 100
    context: str = "notebook"
    plotly_template: str = "plotly_white"
    plotly_colorway: list[str] | None = None
    seaborn_kwargs: dict[str, Any] = field(default_factory=dict)

    def apply_global(self) -> None:
        """Apply plotting style globally (no automatic restore)."""
        palette_colors = sns.color_palette(self.palette)
        sns.set_theme(
            style=self.style,
            palette=palette_colors,
            context=self.context,
            **self.seaborn_kwargs,
        )
        sns.set_theme(font_scale=self.font_scale)
        mpl.rcParams.update(
            {
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "figure.dpi": self.figure_dpi,
                "axes.prop_cycle": mpl.cycler(color=palette_colors),
                "font.family": [self.font_family],
            },
        )

        pio.templates.default = self.plotly_template
        if self.plotly_colorway is not None:
            pio.templates[self.plotly_template].layout.colorway = self.plotly_colorway

    @contextmanager
    def apply(self) -> Generator[None, None, None]:
        """Apply style within a context, restoring previous settings afterwards."""
        keys = [
            "axes.titlesize",
            "axes.labelsize",
            "xtick.labelsize",
            "ytick.labelsize",
            "figure.dpi",
        ]
        prev = {k: mpl.rcParams.get(k) for k in keys}
        prev_prop_cycle = mpl.rcParams.get("axes.prop_cycle")
        prev_font_family = mpl.rcParams.get("font.family")
        prev_plotly_template = pio.templates.default
        prev_plotly_colorway = getattr(
            pio.templates[prev_plotly_template].layout,
            "colorway",
            None,
        )
        target_plotly_colorway = getattr(
            pio.templates[self.plotly_template].layout,
            "colorway",
            None,
        )

        palette_colors = sns.color_palette(self.palette)
        sns.set_theme(
            style=self.style,
            palette=palette_colors,
            context=self.context,
            **self.seaborn_kwargs,
        )
        sns.set_theme(font_scale=self.font_scale)
        mpl.rcParams.update(
            {
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "figure.dpi": self.figure_dpi,
                "axes.prop_cycle": mpl.cycler(color=palette_colors),
                "font.family": [self.font_family],
            },
        )
        pio.templates.default = self.plotly_template
        if self.plotly_colorway is not None:
            pio.templates[self.plotly_template].layout.colorway = self.plotly_colorway
        try:
            yield
        finally:
            pio.templates.default = prev_plotly_template
            pio.templates[prev_plotly_template].layout.colorway = prev_plotly_colorway
            if self.plotly_colorway is not None:
                pio.templates[self.plotly_template].layout.colorway = target_plotly_colorway
            mpl.rcParams.update(prev)
            mpl.rcParams["axes.prop_cycle"] = prev_prop_cycle
            mpl.rcParams["font.family"] = prev_font_family


DEFAULT_PLOT_CFG = PlottingConfig()


def _pretty_label(text: str) -> str:
    """Format labels by replacing underscores and title-casing words."""
    if not text:
        return text
    return text.replace("_", " ").title()


def _save_plotly_fig(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")


def _pca_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected values with ndim=2, got {values.ndim}.")
    values = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(values, full_matrices=False)
    return values @ vt[:2].T


def _pca_2d_with_components(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected values with ndim=2, got {values.ndim}.")
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    proj = centered @ components
    return proj, mean, components


def _histogram_edges(values_list: Iterable[np.ndarray], *, bins: int) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for arr in values_list:
        vals = np.asarray(arr, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            arrays.append(vals)
    if not arrays:
        return np.array([0.0, 1.0], dtype=float)
    return np.histogram_bin_edges(np.concatenate(arrays, axis=0), bins=int(bins))


def _histogram_bar(
    values: np.ndarray,
    *,
    edges: np.ndarray,
    name: str,
    color: str | None = None,
    opacity: float = 0.6,
    log1p_counts: bool = False,
) -> go.Bar:
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    counts, _ = np.histogram(vals, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = np.log1p(counts) if log1p_counts else counts
    marker: dict[str, Any] = {"opacity": opacity}
    if color is not None:
        marker["color"] = color
    return go.Bar(x=centers, y=y, name=_pretty_label(name), marker=marker)


BBOX_EDGE_IDX = np.array(
    [
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
    ],
    dtype=np.int64,
)


def _unit_dir_from_az_el(*, az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    """Convert az/el (LUF: az=atan2(x,z), el=asin(y)) to unit vectors."""
    x = torch.cos(el) * torch.sin(az)
    y = torch.sin(el)
    z = torch.cos(el) * torch.cos(az)
    v = torch.stack([x, y, z], dim=-1)
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(1e-8))


def _flatten_edges_for_plotly(
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.asarray(edges, dtype=float).reshape(-1, 2, 3)
    edges_sep = np.concatenate(
        [edges, np.full((edges.shape[0], 1, 3), np.nan, dtype=float)],
        axis=1,
    )
    flat = edges_sep.reshape(-1, 3)
    return flat[:, 0], flat[:, 1], flat[:, 2]


def _voxel_corners(extent: np.ndarray) -> np.ndarray:
    x_min, x_max, y_min, y_max, z_min, z_max = extent.tolist()
    return np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ],
        dtype=float,
    )


def _pose_first_batch(pose: PoseTW) -> PoseTW:
    if pose.ndim == 1:
        return PoseTW(pose._data.unsqueeze(0))
    if pose.ndim == 2 and pose.shape[0] > 1:
        return PoseTW(pose._data[:1])
    return pose


def _as_pose_tw(pose: PoseTW | torch.Tensor) -> PoseTW:
    """Convert a tensor pose representation into PoseTW if needed."""
    if isinstance(pose, PoseTW):
        return pose
    if torch.is_tensor(pose):
        data = pose
        if data.shape[-1] == 12:
            data = data.view(*data.shape[:-1], 3, 4)
        return PoseTW.from_matrix3x4(data)
    raise TypeError(f"Unsupported pose type: {type(pose)!s}")


def _as_pose_batch(pose: PoseTW | torch.Tensor) -> PoseTW:
    """Ensure poses are batched as ``(B, ..., 12)`` for plotting."""
    pose_tw = _as_pose_tw(pose)
    if pose_tw.ndim == 1:
        return PoseTW(pose_tw._data.unsqueeze(0))
    if pose_tw.ndim == 2 and pose_tw.shape[-1] == 12:
        return PoseTW(pose_tw._data.unsqueeze(0))
    return pose_tw


def _broadcast_pose_batch(pose: PoseTW, *, batch_size: int, name: str) -> PoseTW:
    """Broadcast a single pose to ``batch_size`` if needed."""
    if pose.ndim != 2:
        raise ValueError(f"{name} must have shape (B, 12), got ndim={pose.ndim}.")
    if pose.shape[0] == 1 and batch_size > 1:
        return PoseTW(pose._data.expand(batch_size, 12))
    if pose.shape[0] != batch_size:
        raise ValueError(f"{name} must have batch size 1 or match candidates.")
    return pose


def _centers_rig_from_poses(
    reference_pose_world_rig: PoseTW,
    candidate_poses_world_cam: PoseTW,
) -> torch.Tensor:
    """Compute candidate centers in the reference rig frame."""
    pose_world_cam = _as_pose_batch(candidate_poses_world_cam)
    pose_world_rig = _as_pose_batch(reference_pose_world_rig)
    pose_world_rig = _broadcast_pose_batch(
        pose_world_rig,
        batch_size=int(pose_world_cam.shape[0]),
        name="reference_pose_world_rig",
    )
    pose_rig_cam = pose_world_rig.inverse()[:, None] @ pose_world_cam
    return pose_rig_cam.t


def _candidate_valid_fraction(debug: VinForwardDiagnostics) -> torch.Tensor:
    """Return per-candidate validity fraction.

    VIN v1 exposes per-token validity; VIN v2 only exposes candidate validity.
    """
    token_valid = getattr(debug, "token_valid", None)
    if isinstance(token_valid, torch.Tensor):
        return token_valid.float().mean(dim=-1)

    voxel_valid_frac = getattr(debug, "voxel_valid_frac", None)
    if isinstance(voxel_valid_frac, torch.Tensor):
        return voxel_valid_frac.float()

    candidate_valid = getattr(debug, "candidate_valid", None)
    if isinstance(candidate_valid, torch.Tensor):
        return candidate_valid.float()

    centers = debug.candidate_center_rig_m
    return torch.ones(centers.shape[:-1], dtype=torch.float32, device=centers.device)


def _rotate_points_yaw_cw90(
    points: np.ndarray | torch.Tensor,
    *,
    pose_world_frame: PoseTW | None = None,
    undo: bool = False,
) -> np.ndarray | torch.Tensor:
    """Rotate world points by the UI roll (+Z twist) used in display plots."""
    if isinstance(points, np.ndarray):
        if points.size == 0:
            return points
        pts = torch.as_tensor(points, dtype=torch.float32)
        to_numpy = True
    else:
        pts = points
        to_numpy = False
    if pts.numel() == 0:
        return points

    angle = -np.pi / 2 if undo else np.pi / 2
    c, s = float(np.cos(angle)), float(np.sin(angle))
    r_roll = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        device=pts.device,
        dtype=pts.dtype,
    )
    if pose_world_frame is None:
        rot = PoseTW.from_Rt(
            r_roll,
            torch.zeros(3, device=pts.device, dtype=pts.dtype),
        )
        rotated = rot.transform(pts)
    else:
        pose_rot = rotate_yaw_cw90(pose_world_frame, undo=undo)
        pts_frame = pose_world_frame.inverse().transform(pts)
        rotated = pose_rot.transform(pts_frame)
    return rotated.detach().cpu().numpy() if to_numpy else rotated


def _collect_backbone_evidence_points(
    debug: VinForwardDiagnostics,
    *,
    fields: list[str],
    occ_threshold: float,
    max_points: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if not fields:
        return []

    out = debug.backbone_out
    voxel_fields = {
        "occ_pr": out.occ_pr,
        "occ_input": out.occ_input,
        "counts": out.counts,
    }
    selected = [name for name in fields if voxel_fields.get(name) is not None]
    if not selected:
        return []

    pose = _pose_first_batch(out.t_world_voxel)
    extent = out.voxel_extent.detach().cpu().numpy()
    if extent.ndim == 2:
        extent = extent[0]

    samples: list[tuple[str, np.ndarray, np.ndarray]] = []
    pts_world = out.pts_world
    pts_world_np: np.ndarray | None = None
    if isinstance(pts_world, torch.Tensor):
        pts_world = pts_world.detach().cpu()
        if pts_world.ndim == 3:
            pts_world = pts_world[0]
        if pts_world.ndim == 2 and pts_world.shape[1] == 3:
            pts_world_np = pts_world.numpy()

    for name in selected:
        tensor = voxel_fields[name]
        if tensor is None:
            continue
        field = tensor.detach().cpu()
        if field.ndim == 5:
            field = field[0, 0]
        elif field.ndim == 4:
            field = field[0]
        if name == "occ_pr":
            finite = field[torch.isfinite(field)]
            if finite.numel() > 0:
                min_val = float(finite.min().item())
                max_val = float(finite.max().item())
                if min_val < 0.0 or max_val > 1.0:
                    field = torch.sigmoid(field)
        d, h, w = field.shape

        if name == "counts":
            flat = field.reshape(-1)
            if flat.numel() == 0:
                continue
            topk = min(int(max_points), flat.numel())
            _, idx = torch.topk(flat, k=topk)
            indices = np.stack(np.unravel_index(idx.numpy(), field.shape), axis=-1)
            values = flat[idx].numpy()
        else:
            mask = field > float(occ_threshold)
            indices = np.stack(np.nonzero(mask.numpy()), axis=-1)
            values = field[mask].numpy()
            if indices.size == 0:
                flat = field.reshape(-1)
                if flat.numel() == 0:
                    continue
                topk = min(int(max_points), flat.numel())
                _, idx = torch.topk(flat, k=topk)
                indices = np.stack(
                    np.unravel_index(idx.numpy(), field.shape),
                    axis=-1,
                )
                values = flat[idx].numpy()
            if indices.shape[0] > max_points:
                sel = np.random.choice(indices.shape[0], size=max_points, replace=False)
                indices = indices[sel]
                values = values[sel]

        if indices.size == 0:
            continue

        points_world = None
        if pts_world_np is not None:
            flat_idx = indices[:, 0] * (h * w) + indices[:, 1] * w + indices[:, 2]
            flat_idx = np.clip(flat_idx, 0, pts_world_np.shape[0] - 1)
            points_world = pts_world_np[flat_idx]
        if points_world is None:
            points_world = _voxel_indices_to_world(
                indices,
                pose=pose,
                extent=extent,
                shape=(d, h, w),
            )
        values_np = np.asarray(values)
        finite = np.isfinite(points_world).all(axis=1)
        if values_np.shape[0] == points_world.shape[0]:
            values_np = values_np[finite]
        points_world = points_world[finite]
        if points_world.size == 0:
            continue
        samples.append((name, points_world, values_np))

    return samples


def _segment_trace(
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    color: str,
    name: str,
    width: int = 4,
) -> go.Scatter3d:
    starts = np.asarray(starts, dtype=float).reshape(-1, 3)
    ends = np.asarray(ends, dtype=float).reshape(-1, 3)
    segments = np.stack([starts, ends], axis=1)
    x, y, z = _flatten_edges_for_plotly(segments)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": color, "width": width},
        name=name,
        showlegend=True,
    )


def _line_trace(
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: str,
    name: str,
) -> go.Scatter3d:
    points = np.vstack([start, end])
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        line={"color": color, "width": 6},
        name=name,
        showlegend=True,
    )


def _scatter3d(
    points: np.ndarray,
    *,
    name: str,
    color: str | None = None,
    colorscale: str | None = None,
    values: np.ndarray | None = None,
    size: int = 3,
    opacity: float = 0.8,
) -> go.Scatter3d:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    marker: dict[str, object] = {"size": size, "opacity": opacity}
    if values is not None and colorscale is not None:
        marker["color"] = values
        marker["colorscale"] = colorscale
        marker["colorbar"] = {"title": name}
    elif color is not None:
        marker["color"] = color
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=marker,
        name=name,
        showlegend=True,
    )


def _build_shell_descriptor_figure(
    *,
    u: torch.Tensor,
    f: torch.Tensor,
    r: torch.Tensor,
) -> go.Figure:
    """Build a 3D Plotly figure for one candidate shell descriptor."""
    if u.numel() == 0 or f.numel() == 0 or r.numel() == 0:
        return go.Figure()

    u0 = u[0].detach().cpu().numpy()
    f0 = f[0].detach().cpu().numpy()
    r0 = float(r[0].detach().cpu().item())
    t0 = u0 * r0

    axis_len = 1.2
    lim = 2.0

    fig = go.Figure()

    # Reference axes (LUF).
    axes = [
        (np.array([0.0, 0.0, 0.0]), np.array([axis_len, 0.0, 0.0]), "x (left)"),
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, axis_len, 0.0]), "y (up)"),
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, axis_len]), "z (fwd)"),
    ]
    for start, end, label in axes:
        fig.add_trace(_line_trace(start, end, color="#285f82", name=label))
        fig.add_trace(
            go.Scatter3d(
                x=[end[0]],
                y=[end[1]],
                z=[end[2]],
                mode="text",
                text=[label],
                textposition="top center",
                showlegend=False,
            ),
        )

    fig.add_trace(
        go.Scatter3d(
            x=[t0[0]],
            y=[t0[1]],
            z=[t0[2]],
            mode="markers",
            marker={"size": 6, "color": "#fc5555"},
            name="candidate center",
        ),
    )
    fig.add_trace(_line_trace(np.zeros(3), t0, color="#fc5555", name="t = r·u"))
    fig.add_trace(_line_trace(t0, t0 + u0, color="#285f82", name="u (pos dir)"))
    fig.add_trace(_line_trace(t0, t0 + f0, color="#2a9d8f", name="f (forward)"))

    fig.update_layout(
        title=_pretty_label("Shell descriptor (one candidate): t, r, u, f"),
        scene={
            "aspectmode": "data",
            "xaxis": {"title": _pretty_label("X (left)"), "range": [-lim, lim]},
            "yaxis": {"title": _pretty_label("Y (up)"), "range": [-lim, lim]},
            "zaxis": {"title": _pretty_label("Z (fwd)"), "range": [-lim, lim]},
        },
        legend={"orientation": "h", "y": -0.1},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    return fig


def build_voxel_frame_figure(
    debug: VinForwardDiagnostics,
    *,
    axis_scale: float = 0.5,
) -> go.Figure:
    """Plot voxel grid bounds and voxel axes in world coordinates."""
    pose = _pose_first_batch(debug.backbone_out.t_world_voxel)
    extent = debug.backbone_out.voxel_extent
    extent = extent.detach().cpu().numpy()
    if extent.ndim == 2:
        extent = extent[0]

    corners_vox = _voxel_corners(extent)
    corners_world = (
        pose.transform(
            torch.tensor(corners_vox, dtype=torch.float32, device=pose.t.device),
        )
        .detach()
        .cpu()
        .numpy()
    )
    if corners_world.ndim == 3:
        corners_world = corners_world[0]
    edges = corners_world[BBOX_EDGE_IDX]
    x, y, z = _flatten_edges_for_plotly(edges)

    origin = pose.t[0].detach().cpu().numpy()
    axes = pose.R[0].detach().cpu().numpy()
    axes = axes * float(axis_scale)
    axis_traces = [
        _line_trace(origin, origin + axes[:, 0], color="#285f82", name="voxel x"),
        _line_trace(origin, origin + axes[:, 1], color="#2a9d8f", name="voxel y"),
        _line_trace(origin, origin + axes[:, 2], color="#e76f51", name="voxel z"),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line={"color": "gray", "width": 3},
            name="voxel bounds",
        ),
    )
    for trace in axis_traces:
        fig.add_trace(trace)

    fig.update_layout(
        title=_pretty_label("Voxel grid bounds + axes (world frame)"),
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def build_pose_grid_slices_figure(
    pos_grid: torch.Tensor,
    *,
    axis: Literal["D", "H", "W"],
    index: int,
) -> go.Figure:
    """Plot 2D slices of the normalized position grid (x/y/z channels)."""
    if pos_grid.ndim != 5 or pos_grid.shape[1] != 3:
        raise ValueError(
            f"Expected pos_grid shape (B,3,D,H,W), got {tuple(pos_grid.shape)}.",
        )

    grid = pos_grid[0].detach().cpu().numpy()  # 3 D H W
    d, h, w = grid.shape[1], grid.shape[2], grid.shape[3]
    axis_map = {"D": 1, "H": 2, "W": 3}
    dim = axis_map[axis]
    max_index = [d, h, w][dim - 1] - 1
    idx = int(np.clip(index, 0, max_index))

    if dim == 1:
        slices = grid[:, idx, :, :]  # 3 H W
        x_label, y_label = "H", "W"
    elif dim == 2:
        slices = grid[:, :, idx, :]  # 3 D W
        x_label, y_label = "D", "W"
    else:
        slices = grid[:, :, :, idx]  # 3 D H
        x_label, y_label = "D", "H"

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=tuple(_pretty_label(name) for name in ("pos_x", "pos_y", "pos_z")),
    )
    for col in range(3):
        fig.add_trace(
            go.Heatmap(
                z=slices[col],
                colorscale="RdBu",
                colorbar={"title": _pretty_label("value")},
                showscale=(col == 2),
            ),
            row=1,
            col=col + 1,
        )
        fig.update_xaxes(title_text=_pretty_label(x_label), row=1, col=col + 1)
        fig.update_yaxes(title_text=_pretty_label(y_label), row=1, col=col + 1)

    fig.update_layout(title=_pretty_label(f"Position grid slices (axis {axis}, index {idx})"))
    return fig


def build_pose_grid_pca_figure(
    pos_grid: torch.Tensor,
    *,
    pos_proj: torch.nn.Module,
    max_points: int = 8000,
    color_by: Literal["x", "y", "z", "radius"] = "radius",
    show_axes: bool = True,
    axis_scale: float = 0.5,
) -> go.Figure:
    """Project positional grid embeddings to 2D via PCA."""
    if pos_grid.ndim != 5 or pos_grid.shape[1] != 3:
        raise ValueError(
            f"Expected pos_grid shape (B,3,D,H,W), got {tuple(pos_grid.shape)}.",
        )

    grid = pos_grid[0].permute(1, 2, 3, 0).reshape(-1, 3)
    num_points = int(grid.shape[0])
    if num_points > max_points:
        stride = max(1, num_points // max_points)
        grid = grid[::stride]

    with torch.no_grad():
        emb = pos_proj(grid.to(dtype=torch.float32))
    emb_np = emb.detach().cpu().numpy()
    coords_np = grid.detach().cpu().numpy()
    pca, mean, components = _pca_2d_with_components(emb_np)

    if color_by == "radius":
        color = np.linalg.norm(coords_np, axis=1)
        color_label = "radius"
    else:
        idx = {"x": 0, "y": 1, "z": 2}[color_by]
        color = coords_np[:, idx]
        color_label = color_by

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pca[:, 0],
            y=pca[:, 1],
            mode="markers",
            marker={
                "size": 3,
                "color": color,
                "colorscale": "Viridis",
                "colorbar": {"title": _pretty_label(color_label)},
            },
            showlegend=False,
        ),
    )

    if show_axes:
        axis_pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [axis_scale, 0.0, 0.0],
                [0.0, axis_scale, 0.0],
                [0.0, 0.0, axis_scale],
            ],
            dtype=torch.float32,
            device=grid.device,
        )
        with torch.no_grad():
            axis_enc = pos_proj(axis_pts)
        axis_np = axis_enc.detach().cpu().numpy()
        axis_proj = (axis_np - mean) @ components
        origin = axis_proj[0]
        axis_colors = ["#285f82", "#2a9d8f", "#e76f51"]
        axis_labels = ["x", "y", "z"]
        for idx in range(3):
            end = axis_proj[idx + 1]
            fig.add_trace(
                go.Scatter(
                    x=[origin[0], end[0]],
                    y=[origin[1], end[1]],
                    mode="lines+markers",
                    line={"color": axis_colors[idx], "width": 3},
                    marker={"size": 6},
                    name=f"{axis_labels[idx]} axis",
                    showlegend=True,
                ),
            )
    fig.update_layout(
        title=_pretty_label("Positional embedding PCA (pos_proj)"),
        xaxis_title=_pretty_label("PC1"),
        yaxis_title=_pretty_label("PC2"),
    )
    return fig


def build_pose_vec_histogram(
    pose_vec: torch.Tensor,
    *,
    dim_index: int,
    num_bins: int = 60,
    log1p_counts: bool = False,
) -> go.Figure:
    """Histogram of a single pose vector component."""
    if pose_vec.ndim != 3:
        raise ValueError(
            f"Expected pose_vec shape (B,N,D), got {tuple(pose_vec.shape)}.",
        )

    values = pose_vec[..., dim_index].reshape(-1).detach().cpu().numpy()
    edges = _histogram_edges([values], bins=int(num_bins))
    fig = go.Figure(
        _histogram_bar(
            values,
            edges=edges,
            name="value",
            color="#285f82",
            log1p_counts=log1p_counts,
        ),
    )
    fig.update_layout(
        title=_pretty_label(f"Pose vector component {dim_index} distribution"),
        xaxis_title=_pretty_label("value"),
        yaxis_title=_pretty_label("log1p(count)" if log1p_counts else "count"),
    )
    return fig


def build_pose_enc_pca_figure(
    pose_enc: torch.Tensor,
    *,
    color_values: torch.Tensor | None = None,
    color_label: str = "value",
    max_points: int = 8000,
) -> go.Figure:
    """Project pose encodings to 2D via PCA."""
    enc = pose_enc.reshape(-1, pose_enc.shape[-1]).detach().cpu().numpy()
    if enc.shape[0] > max_points:
        stride = max(1, enc.shape[0] // max_points)
        enc = enc[::stride]
        if color_values is not None:
            color_values = color_values.reshape(-1)[::stride]

    pca = _pca_2d(enc)
    if color_values is None:
        color = np.linalg.norm(enc, axis=1)
        label = "norm"
    else:
        color = color_values.reshape(-1).detach().cpu().numpy()
        label = color_label

    fig = go.Figure(
        go.Scatter(
            x=pca[:, 0],
            y=pca[:, 1],
            mode="markers",
            marker={
                "size": 3,
                "color": color,
                "colorscale": "Viridis",
                "colorbar": {"title": _pretty_label(label)},
            },
            showlegend=False,
        ),
    )
    fig.update_layout(
        title=_pretty_label("Pose encoding PCA"),
        xaxis_title=_pretty_label("PC1"),
        yaxis_title=_pretty_label("PC2"),
    )
    return fig


def build_lff_response_figures(
    pose_encoder: LearnableFourierFeatures,
    *,
    input_dim: int,
    input_range: tuple[float, float],
    num_samples: int = 200,
    max_features: int = 96,
) -> dict[str, go.Figure]:
    """Visualize LFF Fourier features and MLP output along one input dimension."""
    low, high = float(input_range[0]), float(input_range[1])
    xs = torch.linspace(low, high, steps=int(num_samples))
    x = torch.zeros((xs.numel(), pose_encoder.input_dim), dtype=torch.float32)
    x[:, input_dim] = xs

    with torch.no_grad():
        xwr = x @ pose_encoder.Wr.T
        fourier = torch.cat([torch.cos(xwr), torch.sin(xwr)], dim=-1) / math.sqrt(
            float(pose_encoder.fourier_dim),
        )
        fourier_plot = fourier[:, : int(max_features)]
        mlp_out = pose_encoder.mlp(fourier)
        mlp_plot = mlp_out[:, : int(max_features)]

    fourier_np = fourier_plot.detach().cpu().numpy().T
    mlp_np = mlp_plot.detach().cpu().numpy().T
    xs_np = xs.detach().cpu().numpy()

    fig_fourier = go.Figure(
        go.Heatmap(
            z=fourier_np,
            x=xs_np,
            colorscale="RdBu",
            colorbar={"title": _pretty_label("value")},
        ),
    )
    fig_fourier.update_layout(
        title=_pretty_label("LFF Fourier features (pre-MLP)"),
        xaxis_title=_pretty_label(f"input dim {input_dim}"),
        yaxis_title=_pretty_label("feature index"),
    )

    fig_mlp = go.Figure(
        go.Heatmap(
            z=mlp_np,
            x=xs_np,
            colorscale="RdBu",
            colorbar={"title": _pretty_label("value")},
        ),
    )
    fig_mlp.update_layout(
        title=_pretty_label("LFF output (post-MLP)"),
        xaxis_title=_pretty_label(f"input dim {input_dim}"),
        yaxis_title=_pretty_label("feature index"),
    )

    return {
        "lff_response_fourier": fig_fourier,
        "lff_response_mlp": fig_mlp,
    }


def build_lff_empirical_figures(
    pose_vec: torch.Tensor,
    pose_encoder: LearnableFourierFeatures,
    *,
    max_features: int = 96,
    hist_bins: int = 60,
    max_points: int = 8000,
    log1p_counts: bool = False,
) -> dict[str, go.Figure]:
    """Empirical distributions for LFF Fourier + MLP outputs using actual pose_vec."""
    if pose_vec.ndim != 3:
        raise ValueError(
            f"Expected pose_vec shape (B,N,D), got {tuple(pose_vec.shape)}.",
        )

    vec = pose_vec.reshape(-1, pose_vec.shape[-1])
    if vec.shape[0] > max_points:
        stride = max(1, vec.shape[0] // max_points)
        vec = vec[::stride]

    with torch.no_grad():
        xwr = vec @ pose_encoder.Wr.T
        fourier = torch.cat([torch.cos(xwr), torch.sin(xwr)], dim=-1) / math.sqrt(
            float(pose_encoder.fourier_dim),
        )
        mlp_out = pose_encoder.mlp(fourier)

    fourier_np = fourier[:, : int(max_features)].detach().cpu().numpy()
    mlp_np = mlp_out[:, : int(max_features)].detach().cpu().numpy()

    fig_fourier_hist = go.Figure()
    fourier_series = [(f"f{idx}", fourier_np[:, idx]) for idx in range(min(6, fourier_np.shape[1]))]
    fourier_edges = _histogram_edges(
        [vals for _, vals in fourier_series],
        bins=int(hist_bins),
    )
    for name, vals in fourier_series:
        fig_fourier_hist.add_trace(
            _histogram_bar(
                vals,
                edges=fourier_edges,
                name=name,
                log1p_counts=log1p_counts,
            ),
        )
    fig_fourier_hist.update_layout(
        barmode="overlay",
        title=_pretty_label("Empirical LFF Fourier feature distributions"),
        xaxis_title=_pretty_label("value"),
        yaxis_title=_pretty_label("log1p(count)" if log1p_counts else "count"),
    )

    fig_mlp_hist = go.Figure()
    mlp_series = [(f"f{idx}", mlp_np[:, idx]) for idx in range(min(6, mlp_np.shape[1]))]
    mlp_edges = _histogram_edges(
        [vals for _, vals in mlp_series],
        bins=int(hist_bins),
    )
    for name, vals in mlp_series:
        fig_mlp_hist.add_trace(
            _histogram_bar(
                vals,
                edges=mlp_edges,
                name=name,
                log1p_counts=log1p_counts,
            ),
        )
    fig_mlp_hist.update_layout(
        barmode="overlay",
        title=_pretty_label("Empirical LFF MLP output distributions"),
        xaxis_title=_pretty_label("value"),
        yaxis_title=_pretty_label("log1p(count)" if log1p_counts else "count"),
    )

    pca_fourier = _pca_2d(fourier_np)
    fig_fourier_pca = go.Figure(
        go.Scatter(
            x=pca_fourier[:, 0],
            y=pca_fourier[:, 1],
            mode="markers",
            marker={"size": 3, "opacity": 0.6},
            showlegend=False,
        ),
    )
    fig_fourier_pca.update_layout(
        title=_pretty_label("LFF Fourier PCA (empirical)"),
        xaxis_title=_pretty_label("PC1"),
        yaxis_title=_pretty_label("PC2"),
    )

    pca_mlp = _pca_2d(mlp_np)
    fig_mlp_pca = go.Figure(
        go.Scatter(
            x=pca_mlp[:, 0],
            y=pca_mlp[:, 1],
            mode="markers",
            marker={"size": 3, "opacity": 0.6},
            showlegend=False,
        ),
    )
    fig_mlp_pca.update_layout(
        title=_pretty_label("LFF MLP PCA (empirical)"),
        xaxis_title=_pretty_label("PC1"),
        yaxis_title=_pretty_label("PC2"),
    )

    return {
        "lff_empirical_fourier_hist": fig_fourier_hist,
        "lff_empirical_mlp_hist": fig_mlp_hist,
        "lff_empirical_fourier_pca": fig_fourier_pca,
        "lff_empirical_mlp_pca": fig_mlp_pca,
    }


def build_geometry_overview_figure(
    debug: VinForwardDiagnostics,
    *,
    snippet: EfmSnippetView,
    reference_pose_world_rig: PoseTW,
    max_candidates: int = 64,
    show_reference_axes: bool = True,
    show_voxel_axes: bool = True,
    display_rotate_yaw_cw90: bool = False,
    show_scene_bounds: bool = True,
    show_crop_bounds: bool = False,
    show_frustum: bool = False,
    frustum_camera: str = "rgb",
    frustum_frame_indices: list[int] | None = None,
    frustum_scale: float = 1.0,
    show_gt_obbs: bool = False,
    gt_timestamp: int | None = None,
    semidense_mode: str = "off",
    max_sem_points: int = 20000,
    show_trajectory: bool = False,
    mark_first_last: bool = True,
    candidate_pose_mode: Literal["ref_rig", "world_cam", "rig"] = "ref_rig",
    candidate_poses_world_cam: PoseTW | None = None,
    candidate_color_mode: Literal["valid_fraction", "solid", "loss"] = "valid_fraction",
    candidate_color: str = "#ffd966",
    candidate_colorscale: str = "Viridis",
    candidate_loss: torch.Tensor | None = None,
    candidate_frusta_indices: list[int] | None = None,
    candidate_frusta_camera: str = "rgb",
    candidate_frusta_frame_index: int | None = None,
    candidate_frusta_scale: float = 0.5,
    candidate_frusta_color: str = "#ff4d4d",
    candidate_frusta_show_axes: bool = False,
    candidate_frusta_show_center: bool = False,
    backbone_fields: list[str] | None = None,
    backbone_occ_threshold: float = 0.5,
    backbone_max_points: int = 40_000,
    backbone_colorscale: str = "Viridis",
) -> go.Figure:
    """Plot grid bounds, reference/voxel axes, and candidate centers in one figure."""
    builder = SnippetPlotBuilder.from_snippet(
        snippet,
        title=_pretty_label("VIN geometry overview"),
        height=700,
    )

    if show_scene_bounds:
        builder.add_bounds_box(name="Scene bounds", color="gray", dash="dash", width=2)

    if show_crop_bounds and snippet.crop_bounds is not None:
        crop_aabb = tuple(b.detach().cpu().numpy() for b in snippet.crop_bounds)
        builder.add_bounds_box(
            name="Crop bounds",
            color="orange",
            dash="solid",
            width=3,
            aabb=crop_aabb,
        )

    if show_trajectory:
        builder.add_trajectory(mark_first_last=mark_first_last, show=True)

    ref_pose = _pose_first_batch(reference_pose_world_rig)
    if show_reference_axes:
        builder.add_frame_axes(frame=ref_pose, is_rotate_yaw_cw90=False)

    voxel_pose_raw = _pose_first_batch(debug.backbone_out.t_world_voxel)
    voxel_pose = voxel_pose_raw
    if display_rotate_yaw_cw90:
        voxel_pose = rotate_yaw_cw90(voxel_pose_raw)
    if show_voxel_axes:
        builder.add_frame_axes(frame=voxel_pose, is_rotate_yaw_cw90=False)

    if semidense_mode != "off":
        builder.add_semidense(
            max_points=max_sem_points,
            last_frame_only=(semidense_mode == "last frame only"),
        )

    if show_frustum:
        frustum_cam = frustum_camera.replace("-", "")
        builder.add_frusta(
            camera=frustum_cam,
            frame_indices=frustum_frame_indices or [0],
            scale=frustum_scale,
            include_axes=True,
            include_center=True,
            name="Frustum",
        )

    if show_gt_obbs:
        gt_cam = "slamr" if frustum_camera == "slam-r" else frustum_camera
        builder.add_gt_obbs(camera=gt_cam, timestamp=gt_timestamp)

    extent = debug.backbone_out.voxel_extent.detach().cpu().numpy()
    if extent.ndim == 2:
        extent = extent[0]
    corners_vox = _voxel_corners(extent)
    corners_world = (
        voxel_pose.transform(
            torch.tensor(corners_vox, dtype=torch.float32, device=voxel_pose.t.device),
        )
        .detach()
        .cpu()
        .numpy()
    )
    if corners_world.ndim == 3:
        corners_world = corners_world[0]
    edges = corners_world[BBOX_EDGE_IDX]
    x, y, z = _flatten_edges_for_plotly(edges)
    builder.fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line={"color": "#7a7a7a", "width": 3},
            name="voxel grid",
            showlegend=True,
        ),
    )
    builder._update_scene_ranges(corners_world)

    backbone_fields = list(backbone_fields or [])
    if backbone_fields:
        evidence = _collect_backbone_evidence_points(
            debug,
            fields=backbone_fields,
            occ_threshold=backbone_occ_threshold,
            max_points=int(backbone_max_points),
        )
        if evidence:
            bar_len = min(0.25, 0.8 / max(1, len(evidence)))
            bar_gap = min(0.05, 0.2 / max(1, len(evidence)))
            bar_start = 0.95
            for idx, (name, points_world, values) in enumerate(evidence):
                if points_world.size == 0:
                    continue
                if display_rotate_yaw_cw90:
                    points_world = _rotate_points_yaw_cw90(
                        points_world,
                        pose_world_frame=voxel_pose_raw,
                    )
                bar_y = max(0.1, bar_start - idx * (bar_len + bar_gap))
                builder.fig.add_trace(
                    go.Scatter3d(
                        x=points_world[:, 0],
                        y=points_world[:, 1],
                        z=points_world[:, 2],
                        mode="markers",
                        marker={
                            "size": 3,
                            "color": values,
                            "colorscale": backbone_colorscale,
                            "opacity": 0.85,
                            "colorbar": {
                                "title": _pretty_label(name),
                                "x": 1.12,
                                "len": bar_len,
                                "y": bar_y,
                            },
                        },
                        name=f"backbone {name}",
                        showlegend=True,
                    ),
                )
                builder._update_scene_ranges(points_world)

    centers_rig = debug.candidate_center_rig_m.reshape(-1, 3)
    if candidate_poses_world_cam is not None:
        try:
            centers_rig = _centers_rig_from_poses(
                reference_pose_world_rig,
                _as_pose_tw(candidate_poses_world_cam),
            ).reshape(-1, 3)
        except Exception:  # pragma: no cover - fallback to debug centers
            centers_rig = debug.candidate_center_rig_m.reshape(-1, 3)
    valid_frac = _candidate_valid_fraction(debug).reshape(-1)
    centers_world: torch.Tensor | None = None
    if candidate_pose_mode == "world_cam" and candidate_poses_world_cam is not None:
        centers_world = candidate_poses_world_cam.t
    elif candidate_pose_mode == "rig":
        centers_world = centers_rig
    else:
        centers_world = ref_pose.transform(centers_rig.to(ref_pose.t.device))

    if centers_world is not None and centers_world.numel() > 0:
        centers_world = centers_world.reshape(-1, 3)
        valid_frac = valid_frac.reshape(-1)
        color_values = valid_frac
        color_title = "valid frac"
        if candidate_color_mode == "loss" and candidate_loss is not None:
            color_values = candidate_loss.to(device=centers_world.device).reshape(-1)
            color_values = torch.nan_to_num(
                color_values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            color_title = "loss"
        finite = torch.isfinite(centers_world).all(dim=-1)
        centers_world = centers_world[finite]
        valid_frac = valid_frac[finite]
        if candidate_color_mode != "solid":
            color_values = color_values[finite]

        if centers_world.shape[0] > max_candidates:
            idx = torch.randperm(centers_world.shape[0], device=centers_world.device)[:max_candidates]
            centers_world = centers_world[idx]
            valid_frac = valid_frac[idx]
            if candidate_color_mode != "solid":
                color_values = color_values[idx]

        if centers_world.numel() > 0:
            centers_np = centers_world.detach().cpu().numpy()
            if centers_np.ndim != 2:
                centers_np = centers_np.reshape(-1, 3)
            valid_np = valid_frac.detach().cpu().numpy()
            marker: dict[str, object] = {"size": 4, "opacity": 0.85}
            if candidate_color_mode == "solid":
                marker["color"] = candidate_color
            else:
                color_np = color_values.detach().cpu().numpy() if torch.is_tensor(color_values) else valid_np
                marker.update(
                    {
                        "color": color_np,
                        "colorscale": candidate_colorscale,
                        "colorbar": {
                            "title": _pretty_label(color_title),
                            "x": 1.05,
                            "len": 0.7,
                        },
                    },
                )
            builder.fig.add_trace(
                go.Scatter3d(
                    x=centers_np[:, 0],
                    y=centers_np[:, 1],
                    z=centers_np[:, 2],
                    mode="markers",
                    marker=marker,
                    name="candidate centers",
                ),
            )
            builder._update_scene_ranges(centers_np)

    if candidate_frusta_indices and candidate_poses_world_cam is not None:
        cam_label = candidate_frusta_camera.replace("-", "")
        cam_view = snippet.get_camera(cam_label)
        frame_idx = 0 if candidate_frusta_frame_index is None else int(candidate_frusta_frame_index)
        frame_idx = max(0, min(frame_idx, int(cam_view.calib.shape[0]) - 1))
        cam_calib = cam_view.calib[frame_idx]
        pose_data = candidate_poses_world_cam._data[candidate_frusta_indices]
        pose_list = PoseTW(pose_data)
        builder._add_frusta_for_poses(
            cams=cam_calib,
            poses=pose_list,
            scale=candidate_frusta_scale,
            color=candidate_frusta_color,
            name="candidate frusta",
            max_frustums=None,
            include_axes=candidate_frusta_show_axes,
            include_center=candidate_frusta_show_center,
        )

    builder.fig.update_layout(
        scene=dict(aspectmode="data", **builder.scene_ranges),
        height=builder.height,
        legend={"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"},
    )
    return builder.fig


def build_valid_fraction_figure(
    debug: VinForwardDiagnostics,
) -> go.Figure:
    """Plot candidate centers colored by validity fraction."""
    centers = debug.candidate_center_rig_m.reshape(-1, 3).detach().cpu().numpy()
    valid_frac = _candidate_valid_fraction(debug).reshape(-1).detach().cpu().numpy()
    if centers.shape[0] == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        _scatter3d(
            centers,
            values=valid_frac,
            colorscale="Viridis",
            name="valid_frac",
            size=4,
            opacity=0.9,
        ),
    )
    fig.update_layout(
        title=_pretty_label("Candidate centers colored by validity fraction"),
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def build_frustum_samples_figure(
    debug: VinForwardDiagnostics,
    *,
    p3d_cameras: PerspectiveCameras,
    candidate_index: int,
    grid_size: int,
    depths_m: list[float],
) -> go.Figure:
    """Plot frustum sample points colored by token validity (world frame)."""
    points_world_flat = _build_frustum_points_world_p3d(
        p3d_cameras,
        grid_size=int(grid_size),
        depths_m=list(depths_m),
    )
    num_cams = int(points_world_flat.shape[0])

    token_valid = getattr(debug, "token_valid", None)
    if not isinstance(token_valid, torch.Tensor):
        return go.Figure()
    if token_valid.ndim == 3:
        token_valid = token_valid[0]
    if candidate_index >= token_valid.shape[0]:
        return go.Figure()

    if num_cams == 0:
        return go.Figure()
    cam_idx = min(max(int(candidate_index), 0), num_cams - 1)
    points = points_world_flat[cam_idx]

    valid = token_valid[candidate_index].reshape(-1).detach().cpu().numpy()
    points_np = points.detach().cpu().numpy()

    frustum_pose = _pose_from_p3d_camera(p3d_cameras, cam_idx)
    frustum_cam = _camera_tw_from_p3d(p3d_cameras, cam_idx)
    builder = _frustum_builder_stub(
        frustum_pose,
        title=_pretty_label("Frustum samples (world) colored by token validity"),
    )
    fig = builder.fig if builder is not None else go.Figure()
    if points_np.shape[0] > 0:
        fig.add_trace(
            _scatter3d(
                points_np[valid],
                name="valid",
                color="#2a9d8f",
                size=2,
                opacity=0.8,
            ),
        )
        fig.add_trace(
            _scatter3d(
                points_np[~valid],
                name="invalid",
                color="#e76f51",
                size=2,
                opacity=0.4,
            ),
        )
        if builder is not None:
            builder._update_scene_ranges(points_np)

    if builder is not None and frustum_cam is not None and frustum_pose is not None:
        frustum_scale = float(max(depths_m) if depths_m else 1.0)
        builder._add_frusta_for_poses(
            cams=frustum_cam,
            poses=frustum_pose,
            scale=frustum_scale,
            color="#ff4d4d",
            name="candidate frustum",
            max_frustums=None,
            include_axes=False,
            include_center=True,
        )

    fig.update_layout(
        title=_pretty_label("Frustum samples (world) colored by token validity"),
        scene=dict(aspectmode="data", **(builder.scene_ranges if builder is not None else {})),
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


@dataclass(slots=True)
class _FrustumTrajectoryStub:
    t_world_rig: PoseTW


@dataclass(slots=True)
class _FrustumSnippetStub:
    trajectory: _FrustumTrajectoryStub
    mesh: None = None
    semidense: None = None


def _frustum_builder_stub(
    pose_world_cam: PoseTW | None,
    *,
    title: str,
    height: int = 560,
) -> SnippetPlotBuilder | None:
    if pose_world_cam is None:
        return None
    snippet_stub = _FrustumSnippetStub(_FrustumTrajectoryStub(t_world_rig=pose_world_cam))
    return SnippetPlotBuilder.from_snippet(
        snippet_stub,  # type: ignore[arg-type]
        title=title,
        height=height,
    )


def _select_p3d_param(param: torch.Tensor | None, index: int) -> torch.Tensor | None:
    if param is None:
        return None
    if param.ndim == 1:
        return param
    if param.shape[0] == 1:
        return param[0]
    return param[min(max(index, 0), param.shape[0] - 1)]


def _camera_tw_from_p3d(cameras: PerspectiveCameras, index: int) -> CameraTW | None:
    if int(cameras.R.shape[0]) == 0:
        return None
    focal = _select_p3d_param(cameras.focal_length, index)
    principal = _select_p3d_param(cameras.principal_point, index)
    image_size = _select_p3d_param(cameras.image_size, index)
    if focal is None or principal is None or image_size is None:
        return None

    focal = focal.reshape(-1)
    principal = principal.reshape(-1)
    image_size = image_size.reshape(-1)
    if focal.numel() != 2 or principal.numel() != 2 or image_size.numel() != 2:
        return None

    height = image_size[0].reshape(1)
    width = image_size[1].reshape(1)
    params = torch.stack((focal[0], focal[1], principal[0], principal[1]), dim=0)

    return CameraTW.from_surreal(
        width=width,
        height=height,
        type_str="Pinhole",
        params=params,
    )


def _pose_from_p3d_camera(cameras: PerspectiveCameras, index: int) -> PoseTW | None:
    if int(cameras.R.shape[0]) == 0:
        return None
    rot = _select_p3d_param(cameras.R, index)
    trans = _select_p3d_param(cameras.T, index)
    if rot is None or trans is None:
        return None
    rot = rot.reshape(3, 3)
    trans = trans.reshape(3)
    t_wc = -(rot @ trans.unsqueeze(-1)).squeeze(-1)
    return PoseTW.from_Rt(rot, t_wc)


def build_semidense_projection_figure(
    points_world: torch.Tensor,
    *,
    p3d_cameras: PerspectiveCameras,
    candidate_index: int,
    max_points: int = 20000,
    show_frustum: bool = False,
    frustum_scale: float | None = None,
    frustum_color: str = "#ff4d4d",
) -> go.Figure:
    """Plot semidense points colored by whether they project inside the candidate view."""
    if points_world.ndim == 2:
        points_world = points_world.unsqueeze(0)
    if points_world.ndim != 3:
        return go.Figure()

    if points_world.shape[0] > 1:
        points_world = points_world[:1]

    cam = p3d_cameras.to(points_world.device)
    num_cams = int(cam.R.shape[0])
    if num_cams == 0:
        return go.Figure()
    in_ndc = getattr(cam, "in_ndc", False)
    if callable(in_ndc):
        in_ndc = in_ndc()
    needs_rebuild = not hasattr(cam, "_in_ndc")

    def _ensure_batch(param: torch.Tensor | None) -> torch.Tensor | None:
        if param is None:
            return None
        if param.ndim == 1:
            return param.unsqueeze(0)
        return param

    if num_cams > 1 or needs_rebuild:
        idx = min(int(candidate_index), num_cams - 1)
        rot = cam.R[idx : idx + 1] if num_cams > 1 else cam.R
        trans = cam.T[idx : idx + 1] if num_cams > 1 else cam.T
        cam = PerspectiveCameras(
            device=cam.device,
            R=_ensure_batch(rot),
            T=_ensure_batch(trans),
            focal_length=_ensure_batch(cam.focal_length[idx : idx + 1] if num_cams > 1 else cam.focal_length),
            principal_point=_ensure_batch(
                cam.principal_point[idx : idx + 1] if num_cams > 1 else cam.principal_point,
            ),
            image_size=_ensure_batch(cam.image_size[idx : idx + 1] if num_cams > 1 else cam.image_size)
            if cam.image_size is not None
            else None,
            in_ndc=bool(in_ndc),
        )
    cam_idx = 0

    pts_world = points_world[0]
    if pts_world.shape[-1] > 3:
        pts_world = pts_world[..., :3]

    if pts_world.shape[0] > max_points:
        idx = torch.randperm(pts_world.shape[0], device=pts_world.device)[:max_points]
        pts_world = pts_world[idx]

    pts_screen = cam.transform_points_screen(pts_world.unsqueeze(0))
    x, y, z = pts_screen.unbind(dim=-1)
    image_size = cam.image_size
    if image_size is None or image_size.numel() == 0:
        return go.Figure()
    h = image_size[:, 0].unsqueeze(1)
    w = image_size[:, 1].unsqueeze(1)
    finite = torch.isfinite(pts_screen).all(dim=-1)
    valid_mask = finite & (z > 0.0) & (x >= 0.0) & (y >= 0.0) & (x <= (w - 1.0)) & (y <= (h - 1.0))
    valid = valid_mask.squeeze(0).detach().cpu().numpy()
    pts_np = pts_world.detach().cpu().numpy()

    frustum_pose = _pose_from_p3d_camera(cam, cam_idx) if show_frustum else None
    frustum_cam = _camera_tw_from_p3d(cam, cam_idx) if show_frustum else None
    builder = None
    if show_frustum:
        builder = _frustum_builder_stub(
            frustum_pose,
            title=_pretty_label("Semidense points colored by candidate visibility"),
        )
    fig = builder.fig if builder is not None else go.Figure()
    if pts_np.shape[0] > 0:
        fig.add_trace(
            _scatter3d(
                pts_np[valid],
                name="in view",
                color="#2a9d8f",
                size=2,
                opacity=0.8,
            ),
        )
        fig.add_trace(
            _scatter3d(
                pts_np[~valid],
                name="out of view",
                color="#e76f51",
                size=2,
                opacity=0.3,
            ),
        )
        if builder is not None:
            builder._update_scene_ranges(pts_np)

    if builder is not None and frustum_cam is not None and frustum_pose is not None:
        scale = float(frustum_scale) if frustum_scale is not None else 1.0
        if frustum_scale is None and bool(valid_mask.any().item()):
            z_valid = z[valid_mask]
            if z_valid.numel() > 0:
                scale = float(z_valid.mean().item())
        scale = max(0.1, scale)
        builder._add_frusta_for_poses(
            cams=frustum_cam,
            poses=frustum_pose,
            scale=scale,
            color=frustum_color,
            name="candidate frustum",
            max_frustums=None,
            include_axes=False,
            include_center=True,
        )

    fig.update_layout(
        title=_pretty_label("Semidense points colored by candidate visibility"),
        scene=dict(aspectmode="data", **(builder.scene_ranges if builder is not None else {})),
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def build_alignment_figures(
    debug: VinForwardDiagnostics,
    *,
    log1p_counts: bool = False,
) -> dict[str, go.Figure]:
    """Plot alignment histograms between candidates and voxel frame."""
    figs: dict[str, go.Figure] = {}
    cand_forward = debug.candidate_forward_dir_rig.reshape(-1, 3)
    cand_center = debug.candidate_center_dir_rig.reshape(-1, 3)
    view_alignment = debug.view_alignment.reshape(-1).detach().cpu().numpy()

    edges = _histogram_edges([view_alignment], bins=60)
    fig_view = go.Figure(
        _histogram_bar(
            view_alignment,
            edges=edges,
            name="alignment",
            log1p_counts=log1p_counts,
        ),
    )
    fig_view.update_layout(
        title=_pretty_label("View alignment dot(f, -u)"),
        xaxis_title=_pretty_label("dot(f, -u)"),
        yaxis_title=_pretty_label("log1p(count)" if log1p_counts else "count"),
    )
    figs["view_alignment"] = fig_view

    if debug.voxel_forward_dir_rig is not None and debug.voxel_center_dir_rig is not None:
        voxel_forward = debug.voxel_forward_dir_rig.reshape(1, 3)
        voxel_center = debug.voxel_center_dir_rig.reshape(1, 3)
        dot_forward = (cand_forward * voxel_forward).sum(dim=-1).detach().cpu().numpy()
        dot_center = (cand_center * voxel_center).sum(dim=-1).detach().cpu().numpy()

        edges_fwd = _histogram_edges([dot_forward], bins=60)
        fig_fwd = go.Figure(
            _histogram_bar(
                dot_forward,
                edges=edges_fwd,
                name="dot",
                log1p_counts=log1p_counts,
            ),
        )
        fig_fwd.update_layout(
            title=_pretty_label("dot(candidate forward, voxel forward)"),
            xaxis_title=_pretty_label("dot"),
            yaxis_title=_pretty_label("log1p(count)" if log1p_counts else "count"),
        )
        figs["forward_vs_voxel"] = fig_fwd

        edges_center = _histogram_edges([dot_center], bins=60)
        fig_center = go.Figure(
            _histogram_bar(
                dot_center,
                edges=edges_center,
                name="dot",
                log1p_counts=log1p_counts,
            ),
        )
        fig_center.update_layout(
            title=_pretty_label("dot(candidate center dir, voxel center dir)"),
            xaxis_title=_pretty_label("dot"),
            yaxis_title=_pretty_label("log1p(count)" if log1p_counts else "count"),
        )
        figs["center_vs_voxel"] = fig_center

    return figs


def build_prediction_alignment_figure(
    debug: VinForwardDiagnostics,
    *,
    expected_normalized: np.ndarray,
) -> go.Figure:
    """Scatter of predicted score vs view alignment."""
    view_alignment = debug.view_alignment.reshape(-1).detach().cpu().numpy()
    fig = go.Figure(
        data=go.Scatter(
            x=view_alignment,
            y=expected_normalized,
            mode="markers",
            marker={"size": 5, "color": expected_normalized, "colorscale": "Viridis"},
        ),
    )
    fig.update_layout(
        title=_pretty_label("Predicted score vs view alignment"),
        xaxis_title=_pretty_label("dot(f, -u)"),
        yaxis_title=_pretty_label("expected normalized"),
    )
    return fig


def build_field_slice_figures(
    field: torch.Tensor,
    *,
    channel_names: list[str],
    max_channels: int = 4,
    slice_indices: tuple[int, int, int] | None = None,
    title_prefix: str = "field",
) -> dict[str, go.Figure]:
    """Plot XY/XZ/YZ orthogonal slices for selected channels."""
    if field.ndim != 4:
        return {}

    c, d, h, w = field.shape
    if slice_indices is None:
        slice_indices = (d // 2, h // 2, w // 2)
    z_idx, y_idx, x_idx = slice_indices
    z_idx = int(np.clip(z_idx, 0, d - 1))
    y_idx = int(np.clip(y_idx, 0, h - 1))
    x_idx = int(np.clip(x_idx, 0, w - 1))

    figs: dict[str, go.Figure] = {}
    num = min(max_channels, c)
    for i in range(num):
        channel = field[i].detach().cpu().numpy()
        xy = channel[z_idx]
        xz = channel[:, y_idx, :]
        yz = channel[:, :, x_idx]

        title = channel_names[i] if i < len(channel_names) else f"ch{i}"
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=tuple(_pretty_label(ax) for ax in ("XY", "XZ", "YZ")),
        )
        fig.add_trace(go.Heatmap(z=xy, colorscale="Viridis"), row=1, col=1)
        fig.add_trace(go.Heatmap(z=xz, colorscale="Viridis"), row=1, col=2)
        fig.add_trace(go.Heatmap(z=yz, colorscale="Viridis"), row=1, col=3)
        fig.update_layout(title=_pretty_label(f"{title_prefix}:{title}"))
        figs[title] = fig

    return figs


def build_field_token_histograms(
    debug: VinForwardDiagnostics,
    *,
    channel_names: list[str],
    max_channels: int = 4,
    max_samples: int = 200_000,
    log1p_counts: bool = False,
) -> dict[str, go.Figure]:
    """Overlay histograms of field values and sampled token values."""
    field = debug.field
    tokens = debug.tokens
    if field.ndim != 5 or tokens.ndim != 4:
        return {}

    field = field[0]
    tokens = tokens[0]
    num_channels = min(max_channels, field.shape[0])

    figs: dict[str, go.Figure] = {}
    for i in range(num_channels):
        field_vals = field[i].reshape(-1)
        token_vals = tokens[..., i].reshape(-1)

        if field_vals.numel() > max_samples:
            idx = torch.randperm(field_vals.numel())[:max_samples]
            field_vals = field_vals[idx]
        if token_vals.numel() > max_samples:
            idx = torch.randperm(token_vals.numel())[:max_samples]
            token_vals = token_vals[idx]

        label = channel_names[i] if i < len(channel_names) else f"ch{i}"
        field_np = field_vals.detach().cpu().numpy()
        token_np = token_vals.detach().cpu().numpy()
        edges = _histogram_edges([field_np, token_np], bins=60)
        fig = go.Figure()
        fig.add_trace(
            _histogram_bar(
                field_np,
                edges=edges,
                name="field",
                log1p_counts=log1p_counts,
            ),
        )
        fig.add_trace(
            _histogram_bar(
                token_np,
                edges=edges,
                name="tokens",
                log1p_counts=log1p_counts,
            ),
        )
        fig.update_layout(
            barmode="overlay",
            title=_pretty_label(f"Field vs tokens: {label}"),
            xaxis_title=_pretty_label("value"),
            yaxis_title=_pretty_label("log1p(count)" if log1p_counts else "count"),
        )
        figs[label] = fig

    return figs


def _voxel_indices_to_world(
    indices: np.ndarray,
    *,
    pose: PoseTW,
    extent: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    d, h, w = shape
    x_min, x_max, y_min, y_max, z_min, z_max = extent.tolist()

    dx = (x_max - x_min) / w
    dy = (y_max - y_min) / h
    dz = (z_max - z_min) / d

    z_idx, y_idx, x_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    x = x_min + (x_idx + 0.5) * dx
    y = y_min + (y_idx + 0.5) * dy
    z = z_min + (z_idx + 0.5) * dz
    points_vox = torch.tensor(
        np.stack([x, y, z], axis=-1),
        dtype=torch.float32,
        device=pose.t.device,
    )
    points_world = pose.transform(points_vox)
    return points_world.detach().cpu().numpy()


def build_backbone_evidence_figures(
    debug: VinForwardDiagnostics,
    *,
    occ_threshold: float = 0.5,
    max_points: int = 40_000,
) -> dict[str, go.Figure]:
    """Plot sparse voxel evidence as world-space scatter plots."""
    out = debug.backbone_out
    pose = _pose_first_batch(out.t_world_voxel)
    extent = out.voxel_extent.detach().cpu().numpy()
    if extent.ndim == 2:
        extent = extent[0]

    figs: dict[str, go.Figure] = {}
    voxel_fields = {
        "occ_pr": out.occ_pr,
        "occ_input": out.occ_input,
        "counts": out.counts,
    }

    for name, tensor in voxel_fields.items():
        if tensor is None:
            continue
        field = tensor.detach().cpu()
        if field.ndim == 5:
            field = field[0, 0]
        elif field.ndim == 4:
            field = field[0]
        d, h, w = field.shape

        if name == "counts":
            flat = field.reshape(-1)
            if flat.numel() == 0:
                continue
            topk = min(max_points, flat.numel())
            _, idx = torch.topk(flat, k=topk)
            indices = np.stack(np.unravel_index(idx.numpy(), field.shape), axis=-1)
            values = flat[idx].numpy()
        else:
            mask = field > float(occ_threshold)
            indices = np.stack(np.nonzero(mask.numpy()), axis=-1)
            values = field[mask].numpy()
            if indices.shape[0] > max_points:
                sel = np.random.choice(indices.shape[0], size=max_points, replace=False)
                indices = indices[sel]
                values = values[sel]

        if indices.size == 0:
            continue

        points_world = _voxel_indices_to_world(
            indices,
            pose=pose,
            extent=extent,
            shape=(d, h, w),
        )
        fig = go.Figure()
        fig.add_trace(
            _scatter3d(
                points_world,
                name=name,
                values=values if values.size == points_world.shape[0] else None,
                colorscale="Viridis",
                size=2,
                opacity=0.7,
            ),
        )
        fig.update_layout(
            title=_pretty_label(f"Voxel evidence: {name}"),
            scene={"aspectmode": "data"},
        )
        figs[name] = fig

    return figs


def build_scene_field_evidence_figures(
    debug: VinForwardDiagnostics,
    *,
    channel_names: list[str],
    occ_threshold: float = 0.5,
    max_points: int = 40_000,
) -> dict[str, go.Figure]:
    """Plot scene-field channels (field_in) as world-space scatter plots."""
    field_in = debug.field_in
    if field_in.ndim == 5:
        field_in = field_in[0]
    if field_in.ndim != 4:
        return {}

    pose = _pose_first_batch(debug.backbone_out.t_world_voxel)
    extent = debug.backbone_out.voxel_extent.detach().cpu().numpy()
    if extent.ndim == 2:
        extent = extent[0]

    figs: dict[str, go.Figure] = {}
    num_channels = min(field_in.shape[0], len(channel_names))
    for idx in range(num_channels):
        name = channel_names[idx]
        field = field_in[idx].detach().cpu()
        if field.numel() == 0:
            continue

        if name in {
            "occ_pr",
            "occ_input",
        }:
            mask = field > float(occ_threshold)
            indices = np.stack(np.nonzero(mask.numpy()), axis=-1)
            values = field[mask].numpy()
            if indices.shape[0] > max_points:
                sel = np.random.choice(indices.shape[0], size=max_points, replace=False)
                indices = indices[sel]
                values = values[sel]
        else:
            flat = field.reshape(-1)
            if flat.numel() > max_points:
                values, top_idx = torch.topk(flat, k=max_points)
                indices = np.stack(
                    np.unravel_index(top_idx.numpy(), field.shape),
                    axis=-1,
                )
            else:
                values = flat
                all_idx = torch.arange(flat.numel())
                indices = np.stack(
                    np.unravel_index(all_idx.numpy(), field.shape),
                    axis=-1,
                )

        if indices.size == 0:
            continue

        points_world = _voxel_indices_to_world(
            indices,
            pose=pose,
            extent=extent,
            shape=field.shape,
        )
        fig = go.Figure()
        fig.add_trace(
            _scatter3d(
                points_world,
                name=name,
                values=np.asarray(values),
                colorscale="Viridis",
                size=2,
                opacity=0.7,
            ),
        )
        fig.update_layout(
            title=_pretty_label(f"Scene field: {name}"),
            scene={"aspectmode": "data"},
        )
        figs[name] = fig

    return figs


def build_voxel_roundtrip_figure(
    debug: VinForwardDiagnostics,
    *,
    num_points: int = 2000,
    log1p_counts: bool = False,
) -> go.Figure:
    """Check world↔voxel roundtrip residuals."""
    pose = _pose_first_batch(debug.backbone_out.t_world_voxel)
    extent = debug.backbone_out.voxel_extent
    extent = extent.detach().cpu().numpy()
    if extent.ndim == 2:
        extent = extent[0]

    rng = np.random.default_rng(42)
    x_min, x_max, y_min, y_max, z_min, z_max = extent.tolist()
    points_vox = np.stack(
        [
            rng.uniform(x_min, x_max, size=num_points),
            rng.uniform(y_min, y_max, size=num_points),
            rng.uniform(z_min, z_max, size=num_points),
        ],
        axis=-1,
    )
    points_vox_t = torch.tensor(points_vox, dtype=torch.float32, device=pose.t.device)
    points_world = pose.transform(points_vox_t)
    points_vox_rt = pose.inverse().transform(points_world)
    residual = (points_vox_rt - points_vox_t).norm(dim=-1).detach().cpu().numpy()

    edges = _histogram_edges([residual], bins=50)
    fig = go.Figure(
        _histogram_bar(
            residual,
            edges=edges,
            name="residual",
            log1p_counts=log1p_counts,
        ),
    )
    fig.update_layout(
        title=_pretty_label("World↔voxel roundtrip residual (voxel frame)"),
        xaxis_title=_pretty_label("residual"),
        yaxis_title=_pretty_label("log1p(count)" if log1p_counts else "count"),
    )
    return fig


def build_se3_closure_figure(
    candidate_poses_world_cam: PoseTW | torch.Tensor,
    reference_pose_world_rig: PoseTW | torch.Tensor,
    *,
    bins: int = 50,
    log1p_counts: bool = False,
) -> go.Figure:
    """Check closure consistency of T_world_cam = T_world_rig_ref * T_rig_ref_cam."""
    pose_world_cam = _as_pose_tw(candidate_poses_world_cam)
    if pose_world_cam.ndim == 2:
        pose_world_cam = PoseTW(pose_world_cam._data.unsqueeze(0))
    if pose_world_cam.ndim != 3:
        raise ValueError(
            "candidate_poses_world_cam must have shape (N,12) or (B,N,12).",
        )
    batch_size = int(pose_world_cam.shape[0])

    pose_world_rig = _as_pose_tw(reference_pose_world_rig)
    if pose_world_rig.ndim == 1:
        pose_world_rig = PoseTW(pose_world_rig._data.unsqueeze(0))
    if pose_world_rig.ndim != 2:
        raise ValueError(
            "reference_pose_world_rig must have shape (12,) or (B,12).",
        )
    if pose_world_rig.shape[0] == 1 and batch_size > 1:
        pose_world_rig = PoseTW(pose_world_rig._data.expand(batch_size, 12))
    elif pose_world_rig.shape[0] != batch_size:
        raise ValueError("reference_pose_world_rig batch size mismatch.")

    pose_rig_cam = pose_world_rig.inverse()[:, None] @ pose_world_cam
    pose_world_cam_hat = pose_world_rig[:, None] @ pose_rig_cam

    trans_err = (pose_world_cam_hat.t - pose_world_cam.t).norm(dim=-1).detach().cpu().numpy()
    r_err = pose_world_cam_hat.R.transpose(-1, -2) @ pose_world_cam.R
    trace = r_err[..., 0, 0] + r_err[..., 1, 1] + r_err[..., 2, 2]
    cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    rot_err_deg = torch.rad2deg(torch.acos(cos_angle)).detach().cpu().numpy()

    trans_vals = trans_err.reshape(-1)
    rot_vals = rot_err_deg.reshape(-1)
    edges_trans = _histogram_edges([trans_vals], bins=bins)
    edges_rot = _histogram_edges([rot_vals], bins=bins)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=tuple(_pretty_label(name) for name in ("translation residual (m)", "rotation residual (deg)")),
    )
    fig.add_trace(
        _histogram_bar(
            trans_vals,
            edges=edges_trans,
            name="translation",
            log1p_counts=log1p_counts,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _histogram_bar(
            rot_vals,
            edges=edges_rot,
            name="rotation",
            log1p_counts=log1p_counts,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=_pretty_label("SE(3) closure error (chain consistency)"),
        yaxis_title=_pretty_label("log1p(count)" if log1p_counts else "count"),
    )
    fig.update_xaxes(title_text=_pretty_label("residual"), row=1, col=1)
    fig.update_xaxes(title_text=_pretty_label("residual"), row=1, col=2)
    return fig


def build_voxel_inbounds_figure(
    candidate_poses_world_cam: PoseTW | torch.Tensor,
    t_world_voxel: PoseTW,
    voxel_extent: torch.Tensor,
    *,
    bins: int = 50,
) -> go.Figure:
    """Plot candidate in-bounds ratios and normalized coordinate distributions."""
    pose_world_cam = _as_pose_tw(candidate_poses_world_cam)
    centers_world = pose_world_cam.t.reshape(-1, 3)

    pose_world_voxel = _pose_first_batch(t_world_voxel)
    centers_vox = (pose_world_voxel.inverse() * centers_world).detach().cpu()

    extent = voxel_extent.detach().cpu().numpy()
    if extent.ndim == 2:
        extent = extent[0]
    x_min, x_max, y_min, y_max, z_min, z_max = extent.tolist()
    mins = np.array([x_min, y_min, z_min], dtype=float)
    maxs = np.array([x_max, y_max, z_max], dtype=float)
    center = 0.5 * (mins + maxs)
    scale = 0.5 * (maxs - mins)
    scale = np.where(scale <= 1e-6, 1.0, scale)

    coords = centers_vox.numpy()
    normed = (coords - center[None, :]) / scale[None, :]
    axis_in = (coords >= mins[None, :]) & (coords <= maxs[None, :])
    in_all = axis_in.all(axis=1)
    in_euclid = np.linalg.norm(normed, axis=1) <= 1.0

    ratios = [
        axis_in[:, 0].mean(),
        axis_in[:, 1].mean(),
        axis_in[:, 2].mean(),
        in_all.mean(),
        in_euclid.mean(),
    ]
    ratio_labels = ["x in", "y in", "z in", "all axes", "euclidean"]

    edges = _histogram_edges([normed[:, 0], normed[:, 1], normed[:, 2]], bins=bins)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=tuple(_pretty_label(name) for name in ("in-bounds ratios", "normalized coord distributions")),
    )
    fig.add_trace(
        go.Bar(x=ratio_labels, y=ratios, marker={"color": "#2a9d8f"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        _histogram_bar(normed[:, 0], edges=edges, name="x", color="#285f82"),
        row=1,
        col=2,
    )
    fig.add_trace(
        _histogram_bar(normed[:, 1], edges=edges, name="y", color="#2a9d8f"),
        row=1,
        col=2,
    )
    fig.add_trace(
        _histogram_bar(normed[:, 2], edges=edges, name="z", color="#e76f51"),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=_pretty_label("Voxel extent in-bounds ratios + normalized coords"),
        yaxis2_title=_pretty_label("count"),
    )
    fig.update_yaxes(
        range=[0, 1],
        row=1,
        col=1,
        title_text=_pretty_label("fraction"),
    )
    fig.update_xaxes(title_text=_pretty_label("normalized value"), row=1, col=2)
    return fig


def build_pos_grid_linearity_figure(
    pos_grid: torch.Tensor,
    voxel_extent: torch.Tensor,
    *,
    max_points: int = 50_000,
) -> go.Figure:
    """Check linearity between voxel coordinates and pos_grid values."""
    if pos_grid.ndim != 5 or pos_grid.shape[1] != 3:
        raise ValueError(
            f"Expected pos_grid shape (B,3,D,H,W), got {tuple(pos_grid.shape)}.",
        )
    grid = pos_grid[0].detach().cpu().numpy()  # 3 D H W
    d, h, w = grid.shape[1:]

    extent = voxel_extent.detach().cpu().numpy()
    if extent.ndim == 2:
        extent = extent[0]
    x_min, x_max, y_min, y_max, z_min, z_max = extent.tolist()
    xs = np.linspace(x_min, x_max, d)
    ys = np.linspace(y_min, y_max, h)
    zs = np.linspace(z_min, z_max, w)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.stack([xg, yg, zg], axis=-1).reshape(-1, 3)

    pos_vals = grid.transpose(1, 2, 3, 0).reshape(-1, 3)
    total = coords.shape[0]
    if total > max_points:
        stride = max(1, total // max_points)
        coords = coords[::stride]
        pos_vals = pos_vals[::stride]

    x_mat = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
    coeffs = np.zeros((3, 3), dtype=float)
    r2_vals = np.zeros(3, dtype=float)
    rmse_vals = np.zeros(3, dtype=float)
    for idx in range(3):
        y = pos_vals[:, idx]
        beta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
        y_hat = x_mat @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
        coeffs[idx] = beta[:3]
        r2_vals[idx] = r2
        rmse_vals[idx] = rmse

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=tuple(_pretty_label(name) for name in ("linearity R² (rig axes)", "linear map coeffs")),
    )
    fig.add_trace(
        go.Bar(
            x=[_pretty_label(name) for name in ("rig_x", "rig_y", "rig_z")],
            y=r2_vals,
            customdata=rmse_vals,
            marker={"color": "#285f82"},
            hovertemplate="R²=%{y:.4f}<br>RMSE=%{customdata:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=coeffs,
            x=[_pretty_label(name) for name in ("vox_x", "vox_y", "vox_z")],
            y=[_pretty_label(name) for name in ("rig_x", "rig_y", "rig_z")],
            colorscale="RdBu",
            colorbar={"title": _pretty_label("coef")},
        ),
        row=1,
        col=2,
    )
    fig.update_layout(title=_pretty_label("Pos-grid linearity check"))
    fig.update_yaxes(range=[0, 1], row=1, col=1, title_text=_pretty_label("R²"))
    return fig


def _build_sh_components_figure(
    *,
    lmax: int,
    normalization: str,
    n_az: int = 220,
    n_el: int = 110,
) -> go.Figure:
    """Plot a few real SH components as Plotly heatmaps over az/el."""
    az = torch.linspace(-torch.pi, torch.pi, steps=int(n_az))
    el = torch.linspace(-0.5 * torch.pi, 0.5 * torch.pi, steps=int(n_el))
    az_grid, el_grid = torch.meshgrid(az, el, indexing="xy")
    dirs = _unit_dir_from_az_el(az=az_grid, el=el_grid)

    irreps = o3.Irreps.spherical_harmonics(int(lmax))
    y = o3.spherical_harmonics(
        irreps,
        dirs,
        normalize=True,
        normalization=str(normalization),
    )

    lm: list[tuple[int, int]] = []
    for degree in range(int(lmax) + 1):
        for order in range(-degree, degree + 1):
            lm.append((degree, order))

    wanted = [(0, 0), (1, -1), (1, 0), (1, 1)]
    idxs: list[int] = []
    titles: list[str] = []
    for degree, order in wanted:
        if (degree, order) not in lm:
            continue
        idx = lm.index((degree, order))
        idxs.append(idx)
        titles.append(f"Y(l={degree}, m={order}) component {idx}")

    if not idxs:
        return go.Figure()

    cols = 2
    rows = int(np.ceil(len(idxs) / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=tuple(_pretty_label(title) for title in titles),
    )

    az_deg = np.degrees(az.detach().cpu().numpy())
    el_deg = np.degrees(el.detach().cpu().numpy())

    for i, idx in enumerate(idxs):
        r = i // cols + 1
        c = i % cols + 1
        comp = y[..., idx].detach().cpu().numpy().T
        fig.add_trace(
            go.Heatmap(
                x=az_deg,
                y=el_deg,
                z=comp,
                colorscale="RdBu",
                colorbar={"title": _pretty_label("value")},
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text=_pretty_label("azimuth (deg)"), row=r, col=c)
        fig.update_yaxes(title_text=_pretty_label("elevation (deg)"), row=r, col=c)

    fig.update_layout(
        title=_pretty_label(
            f"Real spherical harmonics over directions on S^2 (lmax={lmax})",
        ),
        height=320 * rows,
    )
    return fig


def _build_radius_fourier_figure(
    *,
    r_min: float,
    r_max: float,
    freqs: Iterable[float],
    log1p_counts: bool = False,
) -> go.Figure:
    """Plot sin radius Fourier features for linear r (meters)."""
    r = torch.linspace(float(r_min), float(r_max), steps=500).view(-1, 1)
    r_np = r.squeeze(1).numpy()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=tuple(_pretty_label(name) for name in ("radius r (meters)", "sin(2π ω r)")),
    )

    edges = _histogram_edges([r_np], bins=50)
    fig.add_trace(
        _histogram_bar(
            r_np,
            edges=edges,
            name="r",
            color="#285f82",
            log1p_counts=log1p_counts,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text=_pretty_label("r"), row=1, col=1)
    fig.update_yaxes(
        title_text=_pretty_label("log1p(count)" if log1p_counts else "count"),
        row=1,
        col=1,
    )

    freq_list = list(freqs)
    colors = plotly_colors.sample_colorscale(
        "Viridis",
        np.linspace(0.15, 0.85, len(freq_list)),
    )
    for color, w in zip(colors, freq_list, strict=True):
        y = torch.sin(2.0 * torch.pi * float(w) * r).squeeze(1).numpy()
        fig.add_trace(
            go.Scatter(
                x=r_np,
                y=y,
                mode="lines",
                line={"color": color},
                name=f"ω={w:g}",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text=_pretty_label("r (m)"), row=1, col=2)
    fig.update_yaxes(title_text=_pretty_label("value"), row=1, col=2)
    fig.update_layout(title=_pretty_label("1D Fourier features for radius (linear input)"))
    return fig


def _build_lff_weight_figures(
    *,
    pose_encoder: LearnableFourierFeatures,
) -> dict[str, go.Figure]:
    """Visualize Learnable Fourier Feature projection weights."""
    wr = pose_encoder.Wr.detach().cpu()
    wr_np = wr.numpy()
    fig_wr = go.Figure(
        go.Heatmap(
            z=wr_np,
            colorscale="RdBu",
            colorbar={"title": _pretty_label("value")},
        ),
    )
    fig_wr.update_layout(
        title=_pretty_label("LFF projection weights (Wr)"),
        xaxis_title=_pretty_label("input dim"),
        yaxis_title=_pretty_label("frequency index"),
    )

    norms = torch.linalg.vector_norm(wr, dim=0).numpy()
    fig_norm = go.Figure(
        go.Bar(
            x=list(range(len(norms))),
            y=norms,
        ),
    )
    fig_norm.update_layout(
        title=_pretty_label("LFF Wr column norms (per input dim)"),
        xaxis_title=_pretty_label("input dim"),
        yaxis_title=_pretty_label("norm"),
    )

    return {
        "lff_wr": fig_wr,
        "lff_wr_norms": fig_norm,
    }


def build_vin_encoding_figures(
    debug: VinForwardDiagnostics,
    *,
    lmax: int,
    sh_normalization: str,
    radius_freqs: Iterable[float],
    pose_encoder_lff: LearnableFourierFeatures | None = None,
    include_legacy_sh: bool = True,
    log1p_counts: bool = False,
) -> dict[str, go.Figure]:
    """Build Plotly figures from VIN forward diagnostics."""
    u = debug.candidate_center_dir_rig.reshape(-1, 3)
    f = debug.candidate_forward_dir_rig.reshape(-1, 3)
    r = debug.candidate_radius_m.reshape(-1, 1)

    r_min = float(r.min().item()) if r.numel() else 0.0
    r_max = float(r.max().item()) if r.numel() else 1.0
    if r_min == r_max:
        r_max = r_min + 1e-3

    figs: dict[str, go.Figure] = {}
    figs["shell_descriptor"] = _build_shell_descriptor_figure(u=u, f=f, r=r)
    if pose_encoder_lff is not None:
        figs.update(_build_lff_weight_figures(pose_encoder=pose_encoder_lff))
    if include_legacy_sh:
        figs["sh_components"] = _build_sh_components_figure(
            lmax=int(lmax),
            normalization=str(sh_normalization),
        )
        figs["radius_fourier_features"] = _build_radius_fourier_figure(
            r_min=r_min,
            r_max=r_max,
            freqs=radius_freqs,
            log1p_counts=log1p_counts,
        )
    return figs


def build_candidate_encoding_figures(
    debug: VinForwardDiagnostics,
    *,
    lmax: int,
    sh_normalization: str,
    radius_freqs: Iterable[float],
    pose_encoder_lff: LearnableFourierFeatures | None = None,
    include_legacy_sh: bool = True,
    max_candidates: int = 512,
    max_sh_components: int = 64,
    max_pose_dims: int = 128,
) -> dict[str, go.Figure]:
    """Build Plotly figures for actual candidate encodings."""
    u = debug.candidate_center_dir_rig.reshape(-1, 3)
    f = debug.candidate_forward_dir_rig.reshape(-1, 3)
    r = debug.candidate_radius_m.reshape(-1, 1)
    pose_enc = debug.pose_enc.reshape(-1, debug.pose_enc.shape[-1])

    n = int(u.shape[0])
    if n == 0:
        return {}

    if n > max_candidates:
        idx = torch.randperm(n, device=u.device)[:max_candidates]
        u = u[idx]
        f = f[idx]
        r = r[idx]
        pose_enc = pose_enc[idx]

    figs: dict[str, go.Figure] = {}

    if include_legacy_sh:
        irreps = o3.Irreps.spherical_harmonics(int(lmax))
        u_sh = o3.spherical_harmonics(
            irreps,
            u.to(torch.float32),
            normalize=True,
            normalization=str(sh_normalization),
        )
        f_sh = o3.spherical_harmonics(
            irreps,
            f.to(torch.float32),
            normalize=True,
            normalization=str(sh_normalization),
        )

        if u_sh.shape[1] > max_sh_components:
            u_sh = u_sh[:, :max_sh_components]
            f_sh = f_sh[:, :max_sh_components]

        u_np = u_sh.detach().cpu().numpy()
        f_np = f_sh.detach().cpu().numpy()

        figs["sh_components_u"] = go.Figure(
            go.Heatmap(
                z=u_np,
                colorscale="RdBu",
                colorbar={"title": _pretty_label("value")},
            ),
        )
        figs["sh_components_u"].update_layout(
            title=_pretty_label("SH components for candidate center directions (u)"),
            xaxis_title=_pretty_label("component index"),
            yaxis_title=_pretty_label("candidate index"),
        )

        figs["sh_components_f"] = go.Figure(
            go.Heatmap(
                z=f_np,
                colorscale="RdBu",
                colorbar={"title": _pretty_label("value")},
            ),
        )
        figs["sh_components_f"].update_layout(
            title=_pretty_label("SH components for candidate forward directions (f)"),
            xaxis_title=_pretty_label("component index"),
            yaxis_title=_pretty_label("candidate index"),
        )

        freq_list = list(radius_freqs)
        if freq_list:
            r_f32 = r.to(torch.float32)
            feats: list[torch.Tensor] = []
            for w in freq_list:
                omega = 2.0 * torch.pi * float(w)
                feats.append(torch.sin(omega * r_f32))
                feats.append(torch.cos(omega * r_f32))
            r_feat = torch.cat(feats, dim=-1)
            r_np = r_feat.detach().cpu().numpy()
            figs["radius_fourier_samples"] = go.Figure(
                go.Heatmap(
                    z=r_np,
                    colorscale="RdBu",
                    colorbar={"title": _pretty_label("value")},
                ),
            )
            figs["radius_fourier_samples"].update_layout(
                title=_pretty_label("Radius Fourier features for candidates (sin/cos)"),
                xaxis_title=_pretty_label("feature index"),
                yaxis_title=_pretty_label("candidate index"),
            )

    if pose_encoder_lff is not None:
        pose_vec = getattr(debug, "pose_vec", None)
        if pose_vec is None:
            view_alignment = debug.view_alignment.reshape(-1, 1)
            pose_vec = torch.cat([u, f, r, view_alignment], dim=-1)
        else:
            pose_vec = pose_vec.reshape(-1, pose_vec.shape[-1])
        xwr = pose_vec @ pose_encoder_lff.Wr.T
        fourier = torch.cat([torch.cos(xwr), torch.sin(xwr)], dim=-1) / math.sqrt(
            float(pose_encoder_lff.fourier_dim),
        )
        if fourier.shape[1] > max_pose_dims:
            fourier = fourier[:, :max_pose_dims]
        fourier_np = fourier.detach().cpu().numpy()
        figs["lff_fourier_features"] = go.Figure(
            go.Heatmap(
                z=fourier_np,
                colorscale="RdBu",
                colorbar={"title": _pretty_label("value")},
            ),
        )
        figs["lff_fourier_features"].update_layout(
            title=_pretty_label("LFF Fourier features (pre-MLP)"),
            xaxis_title=_pretty_label("feature index"),
            yaxis_title=_pretty_label("candidate index"),
        )

    if pose_enc.shape[1] > max_pose_dims:
        pose_enc = pose_enc[:, :max_pose_dims]
    enc_np = pose_enc.detach().cpu().numpy()
    figs["pose_encoding"] = go.Figure(
        go.Heatmap(
            z=enc_np,
            colorscale="RdBu",
            colorbar={"title": _pretty_label("value")},
        ),
    )
    figs["pose_encoding"].update_layout(
        title=_pretty_label("Pose encoder output (candidate embeddings)"),
        xaxis_title=_pretty_label("embedding dim"),
        yaxis_title=_pretty_label("candidate index"),
    )

    return figs


def plot_vin_encodings_from_debug(
    debug: VinForwardDiagnostics,
    *,
    out_dir: Path,
    lmax: int,
    sh_normalization: str,
    radius_freqs: Iterable[float],
    file_stem_prefix: str,
    pose_encoder_lff: LearnableFourierFeatures | None = None,
    include_legacy_sh: bool = True,
) -> dict[str, Path]:
    """Generate VIN encoding plots using Plotly and persist them as HTML.

    Args:
        debug: Diagnostics from :meth:`VinModel.forward_with_debug`.
        out_dir: Output directory for saved figures.
        lmax: Maximum SH degree to visualize (legacy plots only).
        sh_normalization: Spherical harmonics normalization mode (legacy plots only).
        radius_freqs: Frequencies for the legacy radius Fourier feature plot.
        file_stem_prefix: Prefix used for output filenames.
        pose_encoder_lff: Optional LFF encoder used for weight visualizations.
        include_legacy_sh: Whether to include legacy SH/Fourier plots.

    Returns:
        Mapping of figure labels to saved HTML paths.
    """
    build_vin_encoding_figures(
        debug,
        lmax=lmax,
        sh_normalization=sh_normalization,
        radius_freqs=radius_freqs,
        pose_encoder_lff=pose_encoder_lff,
        include_legacy_sh=include_legacy_sh,
    )


__all__ = [
    "DEFAULT_PLOT_CFG",
    "PlottingConfig",
    "build_alignment_figures",
    "build_backbone_evidence_figures",
    "build_candidate_encoding_figures",
    "build_field_slice_figures",
    "build_field_token_histograms",
    "build_frustum_samples_figure",
    "build_semidense_projection_figure",
    "build_lff_empirical_figures",
    "build_lff_response_figures",
    "build_pose_enc_pca_figure",
    "build_pose_grid_pca_figure",
    "build_pose_grid_slices_figure",
    "build_pose_vec_histogram",
    "build_pos_grid_linearity_figure",
    "build_prediction_alignment_figure",
    "build_scene_field_evidence_figures",
    "build_se3_closure_figure",
    "build_valid_fraction_figure",
    "build_vin_encoding_figures",
    "build_voxel_inbounds_figure",
    "build_voxel_frame_figure",
    "build_voxel_roundtrip_figure",
    "plot_vin_encodings_from_debug",
]

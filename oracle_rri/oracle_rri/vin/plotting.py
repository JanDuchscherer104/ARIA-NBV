"""VIN plotting helpers for diagnostics and analysis."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]

from ..data.efm_views import EfmSnippetView
from ..data.plotting import SnippetPlotBuilder
from ..utils.frames import rotate_yaw_cw90
from ..utils.plotting import _histogram_overlay, _plot_slice_grid, _to_numpy

Tensor = torch.Tensor


def _parameter_distribution(
    model: torch.nn.Module,
    *,
    trainable_only: bool = True,
) -> pd.DataFrame:
    """Aggregate parameter counts by top-level module name."""
    rows: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        module = name.split(".", 1)[0]
        rows.append({"module": module, "num_params": int(param.numel())})
    if not rows:
        return pd.DataFrame(columns=["module", "num_params"])
    df = pd.DataFrame(rows)
    df = df.groupby("module", as_index=False)["num_params"].sum()
    return df.sort_values("num_params", ascending=False)


def _pretty_label(text: str) -> str:
    """Format labels by replacing underscores and title-casing words."""
    if not text:
        return text
    return text.replace("_", " ").title()


def _pca_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected values with ndim=2, got {values.ndim}.")
    values = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(values, full_matrices=False)
    return values @ vt[:2].T


def _pca_2d_with_components(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _flatten_edges_for_plotly(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _candidate_valid_fraction(debug: Any) -> torch.Tensor:
    """Return per-candidate validity fraction.

    VIN v1 exposes per-token validity; VIN v2/v3 expose candidate validity.
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
    debug: Any,
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
    seg = np.stack([np.asarray(start, dtype=float), np.asarray(end, dtype=float)], axis=0)
    x, y, z = _flatten_edges_for_plotly(seg)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": color, "width": 3},
        name=name,
        showlegend=True,
    )


def _scatter3d(
    points: np.ndarray,
    *,
    name: str,
    color: str | None = None,
    values: np.ndarray | None = None,
    colorscale: str = "Viridis",
    size: int = 3,
    opacity: float = 0.7,
) -> go.Scatter3d:
    pts = np.asarray(points, dtype=float)
    marker: dict[str, Any] = {"size": size, "opacity": opacity}
    if values is not None:
        marker.update(
            {
                "color": np.asarray(values, dtype=float),
                "colorscale": colorscale,
                "colorbar": {"title": _pretty_label(name)},
            }
        )
    elif color is not None:
        marker["color"] = color

    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker=marker,
        name=_pretty_label(name),
        showlegend=True,
    )


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


def _voxel_indices_to_world_from_cache(
    indices: np.ndarray,
    *,
    pose: PoseTW,
    extent: np.ndarray,
    shape: tuple[int, int, int],
    pts_world: torch.Tensor | None,
) -> np.ndarray:
    """Convert voxel indices to world points using cached centers when available."""
    if torch.is_tensor(pts_world):
        pts = pts_world.detach().cpu()
        if pts.ndim == 3:
            pts = pts[0]
        if pts.ndim == 2 and pts.shape[-1] == 3:
            d, h, w = shape
            if pts.numel() == d * h * w * 3:
                pts_grid = pts.reshape(d, h, w, 3)
                sel = pts_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
                return sel.numpy()
    return _voxel_indices_to_world(
        indices,
        pose=pose,
        extent=extent,
        shape=shape,
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


def build_voxel_frame_figure(
    debug: Any,
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
                "size": 6,
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
                "size": 6,
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


def build_lff_empirical_figures(
    pose_vec: torch.Tensor,
    pose_encoder: torch.nn.Module,
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

    if not hasattr(pose_encoder, "Wr") or not hasattr(pose_encoder, "fourier_dim"):
        raise ValueError("pose_encoder must expose Wr and fourier_dim for LFF plotting.")

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
            marker={"size": 6, "opacity": 0.6},
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
            marker={"size": 6, "opacity": 0.6},
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
    debug: Any,
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


def build_valid_fraction_figure(debug: Any) -> go.Figure:
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


def build_backbone_evidence_figures(
    debug: Any,
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

        points_world = _voxel_indices_to_world_from_cache(
            indices,
            pose=pose,
            extent=extent,
            shape=(d, h, w),
            pts_world=getattr(out, "pts_world", None),
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
    debug: Any,
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

        points_world = _voxel_indices_to_world_from_cache(
            indices,
            pose=pose,
            extent=extent,
            shape=field.shape,
            pts_world=getattr(debug.backbone_out, "pts_world", None),
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
    debug: Any,
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
    centers_vox = pose_world_voxel.inverse().transform(centers_world).reshape(-1, 3).detach().cpu()

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


__all__ = [
    "_histogram_overlay",
    "_parameter_distribution",
    "_plot_slice_grid",
    "_to_numpy",
    "build_backbone_evidence_figures",
    "build_field_slice_figures",
    "build_geometry_overview_figure",
    "build_lff_empirical_figures",
    "build_pose_enc_pca_figure",
    "build_pose_grid_pca_figure",
    "build_pose_grid_slices_figure",
    "build_pose_vec_histogram",
    "build_pos_grid_linearity_figure",
    "build_scene_field_evidence_figures",
    "build_se3_closure_figure",
    "build_semidense_projection_figure",
    "build_valid_fraction_figure",
    "build_voxel_frame_figure",
    "build_voxel_inbounds_figure",
    "build_voxel_roundtrip_figure",
]

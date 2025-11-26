"""Plotting helpers for candidate depth renders and debugging."""

from __future__ import annotations

import math
from collections.abc import Iterable

import plotly.graph_objects as go  # type: ignore[import-untyped]
import torch
from efm3d.aria import CameraTW, PoseTW
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
from torch import Tensor

from ..data.plotting import SnippetPlotBuilder


def depth_grid(
    depths: Tensor,
    *,
    titles: Iterable[str] | None = None,
    max_cols: int = 3,
    zmax: float | None = None,
    zfar: float | None = None,
) -> go.Figure:
    """Visualise a batch of depth maps as a Plotly heatmap grid.

    Args:
        depths: Tensor of shape ``(N, H, W)`` in metres.
        titles: Optional iterable of per-depth titles; trimmed/padded to ``N``.
        max_cols: Maximum number of columns in the subplot grid.
        zmax: Optional upper colour limit; defaults to ``depths.max()``.
        zfar: Optional renderer far plane to annotate hit ratio.
    """

    if depths.ndim != 3:
        raise ValueError(f"depth_grid expects (N,H,W) tensor, got shape {tuple(depths.shape)}")

    depth_np = depths.detach().cpu().numpy()
    num = depth_np.shape[0]
    cols = max(1, min(max_cols, num))
    rows = int(math.ceil(num / cols))
    provided_titles = list(titles) if titles is not None else []
    subplot_titles = [provided_titles[i] if i < len(provided_titles) else f"Candidate {i}" for i in range(num)]
    vmax = float(depth_np.max()) if zmax is None else zmax

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    for idx in range(num):
        r = idx // cols + 1
        c = idx % cols + 1
        fig.add_trace(
            go.Heatmap(
                z=depth_np[idx],
                colorscale="Inferno",
                zmin=0.0,
                zmax=vmax,
                showscale=(idx == num - 1),
                colorbar={"title": "Depth (m)"},
            ),
            row=r,
            col=c,
        )

    threshold = zfar if zfar is not None else float(depth_np.max() + 1e-6)
    depths_f = depths.float()
    hit_ratio = float(((depths_f < threshold).float().mean()).item())
    fig.update_layout(
        height=400 * rows,
        width=500 * cols,
        title=f"Candidate depth renders (hit_ratio={hit_ratio:.3f})",
    )
    return fig


def depth_histogram(depths: Tensor, *, bins: int = 50, zfar: float | None = None) -> go.Figure:
    """Histogram of depth values per candidate."""

    if depths.ndim != 3:
        raise ValueError(f"depth_histogram expects (N,H,W) tensor, got {tuple(depths.shape)}")
    depths_np = depths.detach().cpu().numpy()
    num = depths_np.shape[0]
    rows = int(math.ceil(num / 3))
    fig = make_subplots(rows=rows, cols=3, subplot_titles=[f"cand {i}" for i in range(num)])
    for i in range(num):
        r, c = i // 3 + 1, i % 3 + 1
        vals = depths_np[i].reshape(-1)
        if zfar is not None:
            vals = vals[vals < zfar]
        fig.add_trace(go.Histogram(x=vals, nbinsx=bins, name=f"cand {i}", showlegend=False), row=r, col=c)
    fig.update_layout(title="Depth histograms", height=240 * rows)
    return fig


def hit_ratio_bar(depths: Tensor, *, zfar: float) -> go.Figure:
    """Bar chart of per-candidate hit ratios."""

    if depths.ndim != 3:
        raise ValueError(f"hit_ratio_bar expects (N,H,W) tensor, got {tuple(depths.shape)}")
    hits = (depths < zfar).float().mean(dim=(1, 2)).detach().cpu()
    fig = go.Figure(
        go.Bar(x=list(range(hits.numel())), y=hits.tolist(), text=[f"{h:.2f}" for h in hits], textposition="auto")
    )
    fig.update_layout(title=f"Hit ratio per candidate (zfar={zfar})", yaxis_title="hit ratio")
    return fig


class RenderingPlotBuilder(SnippetPlotBuilder):
    """Rendering-focused extensions on top of :class:`SnippetPlotBuilder`."""

    def add_frusta_with_image_plane(
        self,
        poses: PoseTW,
        camera: CameraTW,
        *,
        plane_dist: float = 1.0,
        color: str = "crimson",
        opacity: float = 0.4,
        max_frustums: int = 16,
        name: str = "Rendered frusta",
    ) -> "RenderingPlotBuilder":
        """Add camera frusta and their image-plane rectangles to the 3D scene."""

        pose_tensor = poses.tensor() if isinstance(poses, PoseTW) else poses
        if pose_tensor.ndim == 2:
            pose_tensor = pose_tensor.unsqueeze(0)
        count = min(pose_tensor.shape[0], max_frustums)
        w, h, fx, fy, cx, cy = self._camera_scalar_intrinsics(camera)
        for idx in range(count):
            pose_i = self._pose_from_any(pose_tensor[idx])
            corners_world = self._image_plane_corners_world(
                pose_i, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, dist=plane_dist
            )
            corners_np = corners_world.detach().cpu().numpy()
            center_np = pose_i.t.detach().cpu().numpy()
            # Plane rectangle
            self.fig.add_trace(
                go.Scatter3d(
                    x=corners_np[[0, 1, 2, 3, 0], 0],
                    y=corners_np[[0, 1, 2, 3, 0], 1],
                    z=corners_np[[0, 1, 2, 3, 0], 2],
                    mode="lines",
                    line={"color": color, "width": 2},
                    opacity=opacity,
                    name=f"{name} plane {idx}",
                    showlegend=False,
                )
            )
            # Frustum edges to plane corners
            for corner in corners_np:
                self.fig.add_trace(
                    go.Scatter3d(
                        x=[center_np[0], corner[0]],
                        y=[center_np[1], corner[1]],
                        z=[center_np[2], corner[2]],
                        mode="lines",
                        line={"color": color, "width": 1},
                        opacity=opacity,
                        name=f"{name} frustum {idx}",
                        showlegend=False,
                    )
                )
        return self

    def add_depth_hits(
        self,
        depths: Tensor,
        poses: PoseTW,
        camera: CameraTW,
        *,
        stride: int = 8,
        max_points: int = 20_000,
        color: str = "teal",
        name: str = "Depth hits",
    ) -> "RenderingPlotBuilder":
        """Scatter hit points back-projected from rendered depth maps."""

        if depths.ndim != 3:
            raise ValueError(f"depths must be (N,H,W), got {tuple(depths.shape)}")
        pose_tensor = poses.tensor() if isinstance(poses, PoseTW) else poses
        if pose_tensor.ndim == 2:
            pose_tensor = pose_tensor.unsqueeze(0)
        num = min(depths.shape[0], pose_tensor.shape[0])
        pts_world = []
        for i in range(num):
            pose_i = self._pose_from_any(pose_tensor[i])
            pts_world.append(
                self._backproject_depth(
                    depth=depths[i],
                    pose=pose_i,
                    camera=camera,
                    stride=stride,
                )
            )
        pts_world_t = torch.cat(pts_world, dim=0)
        if pts_world_t.shape[0] > max_points:
            idx = torch.randperm(pts_world_t.shape[0], device=pts_world_t.device)[:max_points]
            pts_world_t = pts_world_t[idx]
        self.fig.add_trace(
            go.Scatter3d(
                x=pts_world_t[:, 0].cpu(),
                y=pts_world_t[:, 1].cpu(),
                z=pts_world_t[:, 2].cpu(),
                mode="markers",
                marker={"size": 2, "color": color, "opacity": 0.6},
                name=name,
            )
        )
        return self

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _camera_scalar_intrinsics(camera: CameraTW) -> tuple[float, float, float, float, float, float]:
        size = camera.size.reshape(-1, 2)[0].float()
        focals = camera.f.reshape(-1, 2)[0].float()
        centers = camera.c.reshape(-1, 2)[0].float()
        return (
            float(size[0].item()),
            float(size[1].item()),
            float(focals[0].item()),
            float(focals[1].item()),
            float(centers[0].item()),
            float(centers[1].item()),
        )

    @staticmethod
    def _image_plane_corners_world(
        pose: PoseTW,
        *,
        w: float,
        h: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist: float,
    ) -> torch.Tensor:
        """Return 4x3 world coords of image-plane corners at distance ``dist``."""

        device = pose.device
        dtype = pose.dtype
        corners_px = torch.tensor(
            [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], device=device, dtype=dtype
        )  # TL, TR, BR, BL
        x = (corners_px[:, 0] - cx) / fx * dist
        y = (corners_px[:, 1] - cy) / fy * dist
        z = torch.full_like(x, dist)
        pts_cam = torch.stack([x, y, z], dim=1)
        pts_world = pose.transform(pts_cam)
        return pts_world

    def _backproject_depth(
        self,
        depth: Tensor,
        pose: PoseTW,
        camera: CameraTW,
        *,
        stride: int,
    ) -> torch.Tensor:
        """Back-project a depth map into world points on a strided grid."""

        h, w = depth.shape
        grid_y = torch.arange(0, h, stride, device=depth.device, dtype=depth.dtype)
        grid_x = torch.arange(0, w, stride, device=depth.device, dtype=depth.dtype)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        depth_s = depth[yy.long(), xx.long()].reshape(-1)
        # Filter out zfar / empty pixels by using a relative threshold close to the max depth.
        depth_max = torch.max(depth)
        mask = torch.isfinite(depth_s) & (depth_s < depth_max * 0.999)
        depth_s = depth_s[mask]
        xx = xx.reshape(-1)[mask]
        yy = yy.reshape(-1)[mask]

        _, _, fx, fy, cx, cy = self._camera_scalar_intrinsics(camera)
        fx_t = torch.tensor(fx, device=depth.device, dtype=depth.dtype)
        fy_t = torch.tensor(fy, device=depth.device, dtype=depth.dtype)
        cx_t = torch.tensor(cx, device=depth.device, dtype=depth.dtype)
        cy_t = torch.tensor(cy, device=depth.device, dtype=depth.dtype)

        x = (xx - cx_t) / fx_t * depth_s
        y = (yy - cy_t) / fy_t * depth_s
        z = depth_s
        pts_cam = torch.stack([x, y, z], dim=1)
        return pose.transform(pts_cam)

    @staticmethod
    def _pose_from_any(pose_item: torch.Tensor | PoseTW) -> PoseTW:
        """Coerce a PoseTW or tensor slice into PoseTW with shape (3,4)."""

        if isinstance(pose_item, PoseTW):
            return pose_item
        tensor = pose_item
        if tensor.ndim == 1 and tensor.shape[0] == 12:
            tensor = tensor.view(3, 4)
        elif tensor.ndim == 2 and tensor.shape[1] == 12:
            tensor = tensor.view(-1, 3, 4)[0]
        elif tensor.ndim == 3 and tensor.shape[-2:] == (3, 4):
            tensor = tensor
        elif tensor.ndim == 2 and tensor.shape == (3, 4):
            tensor = tensor
        else:
            raise ValueError(f"Unexpected pose shape {tuple(tensor.shape)}")
        return PoseTW.from_matrix3x4(tensor)


__all__ = [
    "depth_grid",
    "depth_histogram",
    "hit_ratio_bar",
    "RenderingPlotBuilder",
]

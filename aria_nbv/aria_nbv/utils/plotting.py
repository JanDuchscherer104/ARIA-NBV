"""Shared plotting helpers for Streamlit diagnostics and reports."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib import colormaps
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from .plotly_helpers import flatten_edges_for_plotly

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def depth_to_color(depth: torch.Tensor, *, percentile: float = 99.5) -> np.ndarray:
    """Colorize a depth or distance map for display.

    Args:
        depth: Depth map tensor.
        percentile: Upper percentile used to clamp the color range.

    Returns:
        Rotated uint8 RGB image for display.
    """

    depth_np = depth.detach().cpu().squeeze().numpy()
    finite_mask = np.isfinite(depth_np)
    if not finite_mask.any():
        rgb = np.zeros(depth_np.shape + (3,), dtype=np.uint8)
    else:
        finite_vals = depth_np[finite_mask]
        vmin = max(np.nanmin(finite_vals), 0.0)
        vmax = np.nanpercentile(finite_vals, percentile)
        if vmax <= vmin:
            vmax = vmin + 1e-3
        norm = np.clip((depth_np - vmin) / (vmax - vmin), 0.0, 1.0)
        cmap = colormaps.get_cmap("viridis")
        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    return np.rot90(rgb, k=-1)


class FrameGridBuilder:
    """Builder for simple image grids backed by Plotly subplots."""

    def __init__(self, rows: int, cols: int, *, titles: list[str], height: int, width: int, title: str):
        """Initialize the image-grid figure layout.

        Args:
            rows: Number of grid rows.
            cols: Number of grid columns.
            titles: Subplot titles in row-major order.
            height: Final figure height in pixels.
            width: Final figure width in pixels.
            title: Figure title.
        """

        self.fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, specs=[[{"type": "image"}] * cols] * rows)
        self.height = height
        self.width = width
        self.title = title

    def add_image(self, img: np.ndarray, *, row: int, col: int) -> "FrameGridBuilder":
        """Add one image subplot and return the builder for chaining."""

        self.fig.add_trace(go.Image(z=img), row=row, col=col)
        return self

    def finalize(self) -> go.Figure:
        """Finalize the figure layout and return the Plotly figure."""

        self.fig.update_layout(height=self.height, width=self.width, title_text=self.title)
        return self.fig


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _pretty_label(text: str) -> str:
    """Format labels by replacing underscores and title-casing words."""
    if not text:
        return text
    return text.replace("_", " ").title()


def _pca_2d(values: np.ndarray) -> np.ndarray:
    """Project a 2D feature matrix to its first two PCA directions."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected values with ndim=2, got {values.ndim}.")
    values = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(values, full_matrices=False)
    return values @ vt[:2].T


def _pca_2d_with_components(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a 2D feature matrix and return the PCA mean and components."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected values with ndim=2, got {values.ndim}.")
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    proj = centered @ components
    return proj, mean, components


def _histogram_edges(values_list: list[np.ndarray] | tuple[np.ndarray, ...], *, bins: int) -> np.ndarray:
    """Return shared histogram bin edges across multiple numeric arrays."""
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
    """Build a histogram bar trace using a shared set of bin edges."""
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    counts, _ = np.histogram(vals, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = np.log1p(counts) if log1p_counts else counts
    marker: dict[str, Any] = {"opacity": opacity}
    if color is not None:
        marker["color"] = color
    return go.Bar(x=centers, y=y, name=_pretty_label(name), marker=marker)


def _scalar_to_rgb(
    values: np.ndarray,
    *,
    percentile: float,
    symmetric: bool,
    cmap_name: str = "viridis",
) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape + (3,), dtype=np.uint8)
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(finite), percentile))
        vmin = -vmax
    else:
        lower = max(0.0, 100.0 - percentile)
        vmin = float(np.nanpercentile(finite, lower))
        vmax = float(np.nanpercentile(finite, percentile))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (values - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    cmap = colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    return rgb


def _plot_slice_grid(
    slices: list[np.ndarray],
    *,
    titles: list[str],
    title: str,
    cols: int = 4,
    percentile: float = 99.0,
    symmetric: bool = False,
    cmap_name: str = "viridis",
) -> go.Figure:
    if not slices:
        return go.Figure()
    num = len(slices)
    cols = max(1, min(cols, num))
    rows = int(np.ceil(num / cols))
    builder = FrameGridBuilder(
        rows=rows,
        cols=cols,
        titles=titles,
        height=320 * rows,
        width=320 * cols,
        title=title,
    )
    for idx, arr in enumerate(slices):
        r = idx // cols + 1
        c = idx % cols + 1
        rgb = _scalar_to_rgb(
            arr,
            percentile=percentile,
            symmetric=symmetric,
            cmap_name=cmap_name,
        )
        builder.add_image(rgb, row=r, col=c)
    return builder.finalize()


def _histogram_overlay(
    series: list[tuple[str, np.ndarray]],
    *,
    bins: int,
    title: str,
    xaxis_title: str,
    log1p_counts: bool,
    log_x: bool = False,
    log_x_epsilon: float = 1e-12,
    pretty_label: callable | None = None,
) -> go.Figure:
    all_vals: list[np.ndarray] = []
    for _, values in series:
        vals = np.asarray(values, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            all_vals.append(vals)
    if not all_vals:
        return go.Figure()

    merged = np.concatenate(all_vals, axis=0)
    if log_x:
        merged = merged[np.isfinite(merged)]
        merged_pos = merged[merged > 0.0]
        if merged_pos.size == 0:
            return go.Figure()
        vmin = float(np.nanmin(merged_pos))
        vmax = float(np.nanmax(merged_pos))
        eps = float(max(log_x_epsilon, vmin * 0.5))
        vmin = max(vmin, eps)
        vmax = max(vmax, vmin * (1.0 + 1e-6))
        edges = np.logspace(np.log10(vmin), np.log10(vmax), num=int(bins) + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
        widths = np.diff(edges)
    else:
        edges = np.histogram_bin_edges(merged, bins=int(bins))
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = None

    label_fn = pretty_label if callable(pretty_label) else (lambda x: x)
    fig = go.Figure()
    for name, values in series:
        vals = np.asarray(values, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if log_x:
            vals = np.clip(vals, eps, None)
        counts, _ = np.histogram(vals, bins=edges)
        y = np.log1p(counts) if log1p_counts else counts
        bar_kwargs: dict[str, Any] = {"x": centers, "y": y, "name": label_fn(name), "opacity": 0.6}
        if widths is not None:
            bar_kwargs["width"] = widths
        fig.add_trace(
            go.Bar(**bar_kwargs),
        )
    fig.update_layout(
        barmode="overlay",
        title=label_fn(title),
        xaxis_title=label_fn(xaxis_title),
        yaxis_title=label_fn("log1p(count)" if log1p_counts else "count"),
    )
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def _plot_hist_counts_mpl(
    values: list[float] | np.ndarray,
    *,
    bins: int,
    log_y: bool,
    ax: plt.Axes,
    color: str | None = None,
) -> None:
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    counts, edges = np.histogram(vals, bins=int(bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    y = counts.astype(float)
    if log_y:
        y[y <= 0] = np.nan
    ax.bar(centers, y, width=widths, align="center", color=color, alpha=0.7)
    if log_y:
        ax.set_yscale("log")


__all__ = [
    "FrameGridBuilder",
    "depth_to_color",
    "flatten_edges_for_plotly",
    "histogram_bar",
    "histogram_edges",
    "histogram_overlay",
    "pca_2d",
    "pca_2d_with_components",
    "plot_hist_counts_mpl",
    "plot_slice_grid",
    "pretty_label",
    "scalar_to_rgb",
    "to_numpy",
]

histogram_bar = _histogram_bar
histogram_edges = _histogram_edges
histogram_overlay = _histogram_overlay
pca_2d = _pca_2d
pca_2d_with_components = _pca_2d_with_components
plot_hist_counts_mpl = _plot_hist_counts_mpl
plot_slice_grid = _plot_slice_grid
pretty_label = _pretty_label
scalar_to_rgb = _scalar_to_rgb
to_numpy = _to_numpy

_depth_to_color = depth_to_color
_flatten_edges_for_plotly = flatten_edges_for_plotly

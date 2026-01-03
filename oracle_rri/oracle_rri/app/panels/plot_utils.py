"""Plotting helpers shared across panels."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from matplotlib import colormaps

from ...data.plotting import FrameGridBuilder
from .common import _pretty_label

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


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
) -> go.Figure:
    all_vals: list[np.ndarray] = []
    for _, values in series:
        vals = np.asarray(values, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            all_vals.append(vals)
    if not all_vals:
        return go.Figure()

    edges = np.histogram_bin_edges(np.concatenate(all_vals, axis=0), bins=int(bins))
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = go.Figure()
    for name, values in series:
        vals = np.asarray(values, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        counts, _ = np.histogram(vals, bins=edges)
        y = np.log1p(counts) if log1p_counts else counts
        fig.add_trace(
            go.Bar(x=centers, y=y, name=_pretty_label(name), opacity=0.6),
        )
    fig.update_layout(
        barmode="overlay",
        title=_pretty_label(title),
        xaxis_title=_pretty_label(xaxis_title),
        yaxis_title=_pretty_label("log1p(count)" if log1p_counts else "count"),
    )
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


__all__ = [
    "_histogram_overlay",
    "_parameter_distribution",
    "_plot_hist_counts_mpl",
    "_plot_slice_grid",
    "_scalar_to_rgb",
    "_to_numpy",
]

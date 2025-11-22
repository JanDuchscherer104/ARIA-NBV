"""Plotting helpers for candidate depth renders."""

from __future__ import annotations

import math
from collections.abc import Iterable

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch import Tensor


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

    Returns:
        Plotly figure containing ``N`` heatmaps laid out in a grid.
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

    # Torch-compatible hit ratio (cast bool -> float)
    threshold = zfar if zfar is not None else float(depth_np.max() + 1e-6)
    depths_f = depths.float()
    hit_ratio = float(((depths_f < threshold).float().mean()).item())
    fig.update_layout(
        height=400 * rows,
        width=500 * cols,
        title=f"Candidate depth renders (hit_ratio={hit_ratio:.3f})",
    )
    return fig


__all__ = ["depth_grid"]

"""Small Plotly-specific helpers shared across plotting surfaces.

This module keeps low-level Plotly geometry flattening and trace builders in
one place so plotting-heavy feature modules can share identical helper logic
without re-implementing it or creating import cycles.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]


def flatten_edges_for_plotly(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ``(N, 2, 3)`` edge segments to NaN-separated Plotly XYZ arrays.

    Args:
        edges: Edge array with shape ``(N, 2, 3)`` or any input reshaped to that
            layout.

    Returns:
        Tuple ``(x, y, z)`` ready for a single `Scatter3d` line trace.
    """

    edge_array = np.asarray(edges, dtype=float).reshape(-1, 2, 3)
    separated = np.concatenate(
        [edge_array, np.full((edge_array.shape[0], 1, 3), np.nan, dtype=float)],
        axis=1,
    )
    flat = separated.reshape(-1, 3)
    return flat[:, 0], flat[:, 1], flat[:, 2]


def make_line_trace3d(
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: str,
    name: str,
    width: int = 3,
    showlegend: bool = True,
) -> go.Scatter3d:
    """Build a simple 3D line trace between two endpoints.

    Args:
        start: Start point with shape ``(3,)``.
        end: End point with shape ``(3,)``.
        color: Plotly line color.
        name: Legend label for the trace.
        width: Plotly line width.
        showlegend: Whether the trace should appear in the legend.

    Returns:
        Plotly `Scatter3d` line trace.
    """

    segment = np.stack([np.asarray(start, dtype=float), np.asarray(end, dtype=float)], axis=0)
    x, y, z = flatten_edges_for_plotly(segment)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": color, "width": width},
        name=name,
        showlegend=showlegend,
    )


def make_scatter3d(
    points: np.ndarray,
    *,
    name: str,
    color: str | None = None,
    values: np.ndarray | None = None,
    colorscale: str | None = None,
    colorbar_title: str | None = None,
    size: int = 3,
    opacity: float = 0.7,
    showlegend: bool = True,
) -> go.Scatter3d:
    """Build a 3D scatter trace from point coordinates.

    Args:
        points: Point array with shape ``(N, 3)``.
        name: Legend label for the trace.
        color: Constant marker color when `values` is not provided.
        values: Optional scalar values used for colorscale-based coloring.
        colorscale: Plotly colorscale name used with `values`.
        colorbar_title: Optional colorbar title. Defaults to `name` when
            `values` are provided.
        size: Marker size.
        opacity: Marker opacity.
        showlegend: Whether the trace should appear in the legend.

    Returns:
        Plotly `Scatter3d` marker trace.
    """

    point_array = np.asarray(points, dtype=float).reshape(-1, 3)
    marker: dict[str, object] = {"size": size, "opacity": opacity}
    if values is not None:
        marker["color"] = np.asarray(values, dtype=float)
        marker["colorscale"] = colorscale or "Viridis"
        marker["colorbar"] = {"title": colorbar_title or name}
    elif color is not None:
        marker["color"] = color

    return go.Scatter3d(
        x=point_array[:, 0],
        y=point_array[:, 1],
        z=point_array[:, 2],
        mode="markers",
        marker=marker,
        name=name,
        showlegend=showlegend,
    )


__all__ = ["flatten_edges_for_plotly", "make_line_trace3d", "make_scatter3d"]

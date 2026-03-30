"""Tests for shared plotting helpers."""

import numpy as np

from aria_nbv.utils.plotting import (
    flatten_edges_for_plotly,
    histogram_bar,
    histogram_edges,
    histogram_overlay,
    pca_2d,
    pca_2d_with_components,
    pretty_label,
)


def test_histogram_overlay_log_x_axis() -> None:
    """Enabling log-x should set axis type and keep x values positive."""
    expected_traces = 2
    series = [
        ("a", np.asarray([0.0, 1e-6, 1e-3, 1e-1, 1.0, 10.0], dtype=float)),
        ("b", np.asarray([0.0, 1e-4, 1e-2, 1e-1, 1.0], dtype=float)),
    ]
    fig = histogram_overlay(
        series,
        bins=8,
        title="test",
        xaxis_title="x",
        log1p_counts=False,
        log_x=True,
    )
    if fig.layout.xaxis.type != "log":
        raise AssertionError("Expected a log-x axis when log_x=True.")
    if len(fig.data) != expected_traces:
        msg = f"Expected {expected_traces} traces, got {len(fig.data)}."
        raise AssertionError(msg)
    for trace in fig.data:
        xs = np.asarray(trace.x, dtype=float)
        if not np.all(xs > 0.0):
            raise AssertionError("Expected positive x centers for log-x histogram.")


def test_histogram_overlay_log_x_requires_positive_values() -> None:
    """When no positive finite values exist, return an empty figure."""
    series = [("a", np.asarray([0.0, -1.0, np.nan], dtype=float))]
    fig = histogram_overlay(
        series,
        bins=8,
        title="test",
        xaxis_title="x",
        log1p_counts=False,
        log_x=True,
    )
    if len(fig.data) != 0:
        raise AssertionError(
            "Expected no traces when log_x=True and data has no positive values.",
        )


def test_pretty_label_formats_snake_case() -> None:
    """Shared label formatting should replace underscores and title-case words."""
    if pretty_label("pm_dist_before") != "Pm Dist Before":
        raise AssertionError("Expected snake_case labels to be humanized consistently.")


def test_pca_helpers_return_two_components() -> None:
    """Shared PCA helpers should expose a stable 2D projection contract."""
    values = np.asarray(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [2.0, 2.0, 4.0],
        ],
        dtype=float,
    )
    proj = pca_2d(values)
    proj_with_components, mean, components = pca_2d_with_components(values)
    if proj.shape != (3, 2):
        raise AssertionError(f"Expected 2D projection shape (3, 2), got {proj.shape}.")
    if proj_with_components.shape != (3, 2):
        raise AssertionError("Expected component-aware PCA helper to return a 2D projection.")
    if mean.shape != (1, 3):
        raise AssertionError("Expected PCA mean to preserve the input feature dimension.")
    if components.shape != (3, 2):
        raise AssertionError("Expected PCA components to map input dims to two axes.")


def test_histogram_bar_uses_shared_edges_and_labels() -> None:
    """Histogram helpers should share binning and humanized trace labels."""
    edges = histogram_edges(
        [
            np.asarray([0.0, 0.5, 1.0], dtype=float),
            np.asarray([0.25, 0.75], dtype=float),
        ],
        bins=3,
    )
    trace = histogram_bar(
        np.asarray([0.0, 0.5, 1.0], dtype=float),
        edges=edges,
        name="candidate_valid_frac",
        color="#123456",
        log1p_counts=True,
    )
    if trace.name != "Candidate Valid Frac":
        raise AssertionError("Expected histogram traces to reuse shared label formatting.")
    if len(trace.x) != len(edges) - 1:
        raise AssertionError("Expected one histogram bar center per edge interval.")


def test_flatten_edges_for_plotly_inserts_nan_separators() -> None:
    """3D line-segment flattening should insert NaN separators between edges."""
    xs, ys, zs = flatten_edges_for_plotly(
        np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            dtype=float,
        ),
    )
    if not np.isnan(xs[2]) or not np.isnan(ys[2]) or not np.isnan(zs[2]):
        raise AssertionError("Expected NaN separators after each edge segment.")

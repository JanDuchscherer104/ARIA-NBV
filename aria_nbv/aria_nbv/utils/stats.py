"""Shared statistical helpers used across analysis and app surfaces.

This module owns lightweight numeric helpers that are generic enough to be
reused across multiple features without tying them to a specific panel,
dataset, or experiment surface.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate a linear slope for an ``x``/``y`` series.

    Args:
        x: One-dimensional x-axis values.
        y: One-dimensional y-axis values aligned with ``x``.

    Returns:
        Estimated first-order slope, or ``NaN`` when the series is too short,
        degenerate, or numerically unstable.
    """

    if x.size < 2 or np.allclose(x, x[0]):
        return float("nan")
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:  # pragma: no cover - numerical guard
        return float("nan")


def segment_indices(num: int, frac: float) -> tuple[slice, slice, slice]:
    """Compute early, middle, and late slices for a series.

    Args:
        num: Number of items in the series.
        frac: Fraction of the series used for the early and late segments.

    Returns:
        Tuple of ``(early, mid, late)`` slices.
    """

    size = max(2, int(num * frac))
    early = slice(0, size)
    late = slice(max(num - size, 0), num)
    mid_start = size
    mid_end = max(num - size, mid_start)
    mid = slice(mid_start, mid_end)
    return early, mid, late


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cliff's delta effect size for two samples.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        Cliff's delta, positive when values in `a` tend to exceed values in `b`.
        Returns `NaN` when either sample is empty.
    """

    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a.reshape(-1, 1)
    b = b.reshape(1, -1)
    greater = np.sum(a > b)
    less = np.sum(a < b)
    denom = float(a.size * b.size)
    return float((greater - less) / denom) if denom > 0 else float("nan")


def bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bootstrap the difference between a statistic over two samples.

    Args:
        a: First sample.
        b: Second sample.
        stat_fn: Statistic applied to each bootstrap resample.
        n_boot: Number of bootstrap draws.
        rng: Optional NumPy generator.

    Returns:
        Bootstrap samples of `stat_fn(a) - stat_fn(b)`.
    """

    if a.size == 0 or b.size == 0:
        return np.array([], dtype=float)
    if rng is None:
        rng = np.random.default_rng(0)
    boot = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        a_sample = rng.choice(a, size=a.size, replace=True)
        b_sample = rng.choice(b, size=b.size, replace=True)
        boot[idx] = stat_fn(a_sample) - stat_fn(b_sample)
    return boot


def bootstrap_slope(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bootstrap the linear slope of paired `x`/`y` observations.

    Args:
        x: One-dimensional x-axis values.
        y: One-dimensional y-axis values aligned with `x`.
        n_boot: Number of bootstrap draws.
        rng: Optional NumPy generator.

    Returns:
        Bootstrap samples of the first-order slope.
    """

    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.array([], dtype=float)
    if rng is None:
        rng = np.random.default_rng(0)
    boot = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample_idx = rng.choice(x.size, size=x.size, replace=True)
        boot[idx] = np.polyfit(x[sample_idx], y[sample_idx], 1)[0]
    return boot


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation without requiring SciPy.

    Args:
        x: One-dimensional x-axis values.
        y: One-dimensional y-axis values aligned with `x`.

    Returns:
        Spearman rank correlation, or `NaN` for empty, mismatched, or degenerate
        inputs.
    """

    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    rank_x = pd.Series(x).rank().to_numpy()
    rank_y = pd.Series(y).rank().to_numpy()
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        return float("nan")
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


__all__ = [
    "bootstrap_diff",
    "bootstrap_slope",
    "cliffs_delta",
    "linear_slope",
    "segment_indices",
    "spearman_rho",
]

"""Regression tests for shared statistical helpers."""

from __future__ import annotations

import math

import numpy as np

from aria_nbv.app.panels.common import linear_slope as panel_linear_slope
from aria_nbv.app.panels.common import segment_indices as panel_segment_indices
from aria_nbv.utils.stats import (
    bootstrap_diff,
    bootstrap_slope,
    cliffs_delta,
    linear_slope,
    segment_indices,
    spearman_rho,
)
from aria_nbv.utils.wandb_utils import linear_slope as wandb_linear_slope
from aria_nbv.utils.wandb_utils import segment_indices as wandb_segment_indices


def test_linear_slope_estimates_expected_gradient() -> None:
    x = np.arange(5, dtype=float)
    y = 2.0 * x + 1.0

    assert math.isclose(linear_slope(x, y), 2.0)


def test_linear_slope_returns_nan_for_degenerate_series() -> None:
    x = np.ones(4, dtype=float)
    y = np.arange(4, dtype=float)

    assert math.isnan(linear_slope(x, y))


def test_segment_indices_covers_early_mid_late_windows() -> None:
    early, mid, late = segment_indices(10, 0.2)

    assert early == slice(0, 2)
    assert mid == slice(2, 8)
    assert late == slice(8, 10)


def test_panel_and_wandb_modules_share_stats_helpers() -> None:
    assert panel_linear_slope is linear_slope
    assert panel_segment_indices is segment_indices
    assert wandb_linear_slope is linear_slope
    assert wandb_segment_indices is segment_indices


def test_cliffs_delta_reports_directional_effect_size() -> None:
    a = np.array([3.0, 4.0, 5.0], dtype=float)
    b = np.array([0.0, 1.0, 2.0], dtype=float)

    assert cliffs_delta(a, b) > 0


def test_bootstrap_helpers_return_requested_sample_count() -> None:
    rng = np.random.default_rng(0)
    a = np.array([1.0, 2.0, 3.0], dtype=float)
    b = np.array([0.0, 1.0, 2.0], dtype=float)
    x = np.arange(5, dtype=float)
    y = 2.0 * x + 1.0

    diff = bootstrap_diff(a, b, stat_fn=np.mean, n_boot=8, rng=rng)
    slope = bootstrap_slope(x, y, n_boot=8, rng=rng)

    assert diff.shape == (8,)
    assert slope.shape == (8,)


def test_spearman_rho_matches_monotone_series() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    y = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)

    assert math.isclose(spearman_rho(x, y), 1.0)

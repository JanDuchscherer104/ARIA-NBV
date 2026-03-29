"""Tests for the panels dispatcher module."""

from __future__ import annotations

from oracle_rri.app import panels
from oracle_rri.app.panels import candidates, data, depth, offline_stats, rri, rri_binning, vin_diagnostics, wandb


def test_panels_dispatcher_reexports() -> None:
    """Ensure dispatcher re-exports dedicated panel renderers."""
    assert panels.render_candidates_page is candidates.render_candidates_page
    assert panels.render_data_page is data.render_data_page
    assert panels.render_depth_page is depth.render_depth_page
    assert panels.render_offline_stats_page is offline_stats.render_offline_stats_page
    assert panels.render_rri_page is rri.render_rri_page
    assert panels.render_rri_binning_page is rri_binning.render_rri_binning_page
    assert panels.render_vin_diagnostics_page is vin_diagnostics.render_vin_diagnostics_page
    assert panels.render_wandb_analysis_page is wandb.render_wandb_analysis_page

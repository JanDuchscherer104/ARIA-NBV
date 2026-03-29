"""Dispatcher for Streamlit panel renderers.

This module intentionally re-exports the dedicated panel modules in
``aria_nbv.app.panels``. Panel logic and plotting helpers live in their
respective component modules (e.g., ``data.plotting`` or
``pose_generation.plotting``).
"""

from __future__ import annotations

from .panels.candidates import render_candidates_page
from .panels.data import render_data_page
from .panels.depth import render_depth_page
from .panels.offline_stats import render_offline_stats_page
from .panels.optuna_sweep import render_optuna_sweep_page
from .panels.rri import render_rri_page
from .panels.rri_binning import render_rri_binning_page
from .panels.testing_attribution import render_testing_attribution_page
from .panels.vin_diagnostics import render_vin_diagnostics_page
from .panels.wandb import render_wandb_analysis_page

__all__ = [
    "render_candidates_page",
    "render_data_page",
    "render_depth_page",
    "render_offline_stats_page",
    "render_optuna_sweep_page",
    "render_rri_page",
    "render_rri_binning_page",
    "render_testing_attribution_page",
    "render_vin_diagnostics_page",
    "render_wandb_analysis_page",
]

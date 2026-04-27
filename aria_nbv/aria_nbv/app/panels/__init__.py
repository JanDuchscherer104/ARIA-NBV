"""Streamlit panels for the NBV app.

Keep panel imports lazy so optional heavy dependencies do not break unrelated
pages at app startup.
"""

from __future__ import annotations

__all__ = [
    "render_candidates_page",
    "render_data_page",
    "render_depth_page",
    "render_offline_stats_page",
    "render_rri_binning_page",
    "render_rri_page",
    "render_vin_diagnostics_page",
    "render_wandb_analysis_page",
    "render_testing_attribution_page",
]


def __getattr__(name: str):
    if name == "render_candidates_page":
        from .candidates import render_candidates_page

        return render_candidates_page
    if name == "render_data_page":
        from .data import render_data_page

        return render_data_page
    if name == "render_depth_page":
        from .depth import render_depth_page

        return render_depth_page
    if name == "render_offline_stats_page":
        from .offline_stats import render_offline_stats_page

        return render_offline_stats_page
    if name == "render_rri_binning_page":
        from .rri_binning import render_rri_binning_page

        return render_rri_binning_page
    if name == "render_rri_page":
        from .rri import render_rri_page

        return render_rri_page
    if name == "render_vin_diagnostics_page":
        from .vin_diagnostics import render_vin_diagnostics_page

        return render_vin_diagnostics_page
    if name == "render_wandb_analysis_page":
        from .wandb import render_wandb_analysis_page

        return render_wandb_analysis_page
    if name == "render_testing_attribution_page":
        from .testing_attribution import render_testing_attribution_page

        return render_testing_attribution_page
    raise AttributeError(name)

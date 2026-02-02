# Panels helpers audit (panels.py vs panels/*)

Date: 2026-01-26

## Redundant helpers in `oracle_rri/oracle_rri/app/panels.py`
These are already implemented elsewhere and should be removed from `panels.py`
in favor of imports.

### Shared UI helpers (use `app/panels/common.py`)
- `_strip_ansi`, `_pretty_label`, `_info_popover`, `_report_exception`

### Plot utilities (moved out of panels)
- `oracle_rri/utils/plotting.py`: `_to_numpy`, `_scalar_to_rgb`, `_plot_slice_grid`,
  `_histogram_overlay`, `_plot_hist_counts_mpl`
- `oracle_rri/vin/plotting.py`: `_to_numpy`, `_plot_slice_grid`,
  `_histogram_overlay`, `_parameter_distribution`
- `oracle_rri/rri_metrics/plotting.py`: `_histogram_overlay`, `_plot_hist_counts_mpl`,
  `rri_color_map`, `plot_rri_scores`, `plot_pm_distances`, `plot_pm_accuracy`,
  `plot_pm_completeness`, `plot_rri_scene`

### Data page helpers (use `app/panels/data.py`)
- `Scene3DPlotOptions`, `pose_world_cam`, `semidense_points_for_frame`,
  `scene_plot_options_ui`

### Candidate helpers (use `app/panels/candidates.py`)
- `rejected_pose_tensor`

### Offline cache helpers (use `app/panels/offline_cache_utils.py`)
- `_load_efm_snippet_for_cache`, `_prepare_offline_cache_dataset`,
  `_collect_offline_cache_stats`

### VIN diagnostics helpers (use `app/panels/vin_utils.py`)
- `_build_experiment_config`, `_run_vin_debug`, `_vin_oracle_batch_from_cache`

### W&B helpers (use `configs/wandb_config.py` + `app/panels/wandb.py`)
- `_load_wandb_history`, `_metric_pairs`, `_select_metric_key`
- `_wandb_run_candidates`, `_resolve_wandb_run`, `_wandb_media_path`,
  `_wandb_media_paths`, `_wandb_download_media` (should live with `panels/wandb.py`
  if still needed; currently unused outside `panels.py`)

### RRI binning helpers (prefer `RriOrdinalBinner`)
- `_load_rri_fit_data` → `RriOrdinalBinner.load_fit_data`
- `_load_binner_data` → `RriOrdinalBinner.load`

## Page renderers duplicated in `panels.py`
Each has a dedicated module under `app/panels/` and should be routed there:
- `render_data_page` → `app/panels/data.py`
- `render_candidates_page` → `app/panels/candidates.py`
- `render_depth_page` → `app/panels/depth.py`
- `render_rri_page` → `app/panels/rri.py`
- `render_vin_diagnostics_page` → `app/panels/vin_diagnostics.py`
- `render_rri_binning_page` → `app/panels/rri_binning.py`
- `render_wandb_analysis_page` → `app/panels/wandb.py`
- `render_offline_stats_page` → `app/panels/offline_stats.py`
- `render_optuna_sweep_page` → `app/panels/optuna_sweep.py`
- `render_testing_attribution_page` → `app/panels/testing_attribution.py`

## Suggested cleanup path
1) Convert `panels.py` into a thin dispatcher importing the panel modules.
2) Delete redundant helper implementations; rely on shared modules above.
3) If W&B run/media helpers are still needed, move them into `app/panels/wandb.py`
   or `configs/wandb_config.py` and import there.

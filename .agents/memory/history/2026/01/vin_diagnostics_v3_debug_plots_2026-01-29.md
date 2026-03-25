---
id: 2026-01-29_vin_diagnostics_v3_debug_plots_2026-01-29
date: 2026-01-29
title: "Vin Diagnostics V3 Debug Plots 2026 01 29"
status: legacy-imported
topics: [diagnostics, v3, debug, plots, 2026]
source_legacy_path: ".codex/vin_diagnostics_v3_debug_plots_2026-01-29.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VINv3 diagnostics: richer summaries + higher-signal plots (2026-01-29)

## Goal
Improve the VIN diagnostics dashboard for **VinModelV3** and make `summarize_vin_v3` surface the most important signals from:
- `VinV3ForwardDiagnostics`
- `VinPrediction`

## Changes made

### 1) `summarize_vin_v3` now exposes more actionable signals
File: `oracle_rri/oracle_rri/vin/summarize_v3.py`

- Adds **metrics**:
  - `candidate_valid_rate`
  - radius quantiles (`candidate_radius_m`)
  - quantiles for `voxel_valid_frac` and `semidense_candidate_vis_frac` (when present)
  - CORAL `monotonicity_violation_rate` and `entropy` quantiles
  - Pearson/Spearman between `oracle_rri` and `expected_normalized` (when oracle labels exist)
- Adds **continuous expected RRI** (`expected_rri`) when bin values are initialized (`head_coral.has_bin_values`).
- Adds more **input inspection** for `VinSnippetView`:
  - number of finite points vs padded points
  - quantiles for `inv_dist_std` and `obs_count` channels when present
- Includes missing debug tensors in the summary (`semidense_grid_feat`) and more stats for large blocks (`field_in`, `field`, `global_feat`, `feats`, `pos_grid`).
- Appends a small **candidate ranking table** (top/bottom by `expected_normalized`) including proxies + radius + oracle RRI (if available).

### 2) Summary tab: candidate leaderboard + proxy correlations
File: `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/summary.py`

- Adds an expander with:
  - **Top‑k leaderboard** rows (`expected_norm`, optional `expected_rri`, `oracle_rri`, proxies, radius).
  - **Correlation table** and scatter plots for:
    - `expected` vs `voxel_valid_frac`
    - `expected` vs `semidense_candidate_vis_frac`
  - Per-feature correlation bars for the 5‑D `semidense_proj` and `voxel_proj` vectors (when present).
- Prefers `cfg.module_config.vin.scene_field_channels` for channel labels (avoids v2/v3 name mismatch).

### 3) Tokens tab: v3 scalars + CNN norm diagnostics + consistent batching
File: `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/tokens.py`

- Adds a single **batch index** selector at the top and uses it consistently (tokens/frusta/projection maps).
- Adds **VINv3 projection scalars** (JSON) for the selected candidate:
  - `semidense_proj` (coverage, empty_frac, semidense_candidate_vis_frac, depth_mean, depth_std)
  - `voxel_proj` (same schema when present)
- Adds **semidense CNN feature norm histogram** (`semidense_grid_feat`) to catch collapse/dead branch issues.

### 4) Encodings + Transforms tabs: pos_grid plots work for v3
Files:
- `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/encodings.py`
- `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/transforms.py`

- Uses `debug.pos_grid` directly when available (v3), falling back to legacy computation when not.

### 5) Geometry: color candidates by predicted score / oracle / proxies
Files:
- `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/geometry.py`
- `oracle_rri/oracle_rri/vin/plotting.py`

- Geometry UI now supports candidate coloring by:
  - predicted score (`expected_normalized`)
  - `oracle_rri`
  - `voxel_valid_frac`
  - `semidense_candidate_vis_frac`
  - CORAL loss (if binner + oracle labels exist)
- `build_geometry_overview_figure(...)` now supports a generic `"scalar"` color mode with:
  - `candidate_color_values`
  - `candidate_color_title`

## Verification
- `ruff format` and `ruff check` on touched files
- `pytest -q oracle_rri/tests/vin/test_vin_model_v3_methods.py`

## Suggested follow-ups (not implemented)
- **Feature ablations in-dash**: toggle zeroing blocks (pose/global/proj/CNN/traj) and re-run forward to see score sensitivity.
- **Attention visibility**:
  - expose `PoseConditionedGlobalPool` attention weights (pose queries ↔ pooled voxels)
  - expose `traj_attn` attention weights when `use_traj_encoder=True`
- **Per-candidate head-input breakdown**:
  - norms and PCA of each block (pose_enc, global_feat, semidense_proj, semidense_grid_feat, traj_ctx)
  - correlation of each block’s norm with `expected_normalized`
- **“Compare two candidates” view**: side-by-side scalar features + projection maps + predicted distributions.

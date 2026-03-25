# VIN Coverage-Aware Training Notes (2026-01-05)

## Findings
- Removing coverage-weighted loss exposed VIN v2 to low-evidence candidates early, flattening CORAL loss slopes.
- Semidense projection validity was not explicitly encoded; invalid/visible points were indistinguishable.
- OneCycleLR runs with low initial LR further slowed early loss changes (confirmed via LR logs).

## Changes Implemented
- Added semidense per-point observation counts in `EfmPointsView.collapse_points` and threaded through VIN snippet caches.
- Added visibility-conditioned embedding for semidense frustum tokens.
- Split validity scalars into `voxel_valid_frac` and `semidense_valid_frac` and logged their stats.
- Added coverage-aware loss weighting with configurable annealing in `VinLightningModule`.
- Expanded grad norm logging for additional VIN submodules.
- Updated `.configs/offline_only.toml` to enable coverage weighting, obs-count features, and semidense frustum tokens.
- Added ablation toggles for `use_point_encoder` / `use_traj_encoder` in `VinModelV2Config`.
- Added one-step cache config: `.configs/offline_cache_required_one_step_vin_cache.toml`.
- Added `vin_snippet_cache_allow_subset` to let VIN training filter offline cache entries to the subset present in `vin_snippet_cache`.

## Suggestions / Next Steps
- Rebuild VIN snippet cache after setting `include_obs_count=true` to ensure cache hashes match.
- `nbv-cache-vin-snippets` now uses `semidense_include_obs_count` from the config; avoid putting `include_obs_count` under `vin_snippet_cache` (not a valid field).
- If PointNeXt should ingest obs-count features, update its `in_channels` to at least 5.
- Use `vin_snippet_cache_allow_subset=true` when the VIN snippet cache is incomplete but you still want to train from the offline cache.
- Validate training on real data for a few epochs to confirm loss slopes improve with annealed coverage weighting.
- Consider a short ablation sweep on:
  - `coverage_weight_strength_start` + `coverage_weight_anneal_epochs`
  - `use_voxel_valid_frac_feature` (True/False)
  - `semidense_visibility_embed` (True/False)

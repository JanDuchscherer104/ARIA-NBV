# Minimal VIN snippet for batching (offline cache)

## Summary
- Added `VinSnippetView` (points_world + t_world_rig) to provide a batchable snippet representation.
- Offline cache dataset now builds this minimal snippet for `return_format="vin_batch"` by collapsing semidense points once in the dataset and returning trajectory poses only.
- Collate function now batches `VinSnippetView` by padding points with NaNs and trajectory poses with the last frame.
- VIN v2 updated to accept `VinSnippetView` and to compute valid fractions using per-camera finite-point counts (padding-safe).

## Key changes
- `oracle_rri/oracle_rri/data/efm_views.py`: new `VinSnippetView` + exports.
- `oracle_rri/oracle_rri/data/offline_cache.py`: compute collapsed semidense points in dataset (`_build_vin_snippet`).
- `oracle_rri/oracle_rri/lightning/lit_datamodule.py`: collate supports minimal snippets; removed batch-size block.
- `oracle_rri/oracle_rri/vin/model_v2.py`: handle `VinSnippetView` in semidense + trajectory paths; valid_frac uses finite count to ignore padded NaNs.
- `oracle_rri/oracle_rri/lightning/lit_module.py`: pass minimal snippet to VIN v2; require backbone_out when no full EFM snippet.

## Limitations
- Full `EfmSnippetView` batching remains unsupported; only `VinSnippetView` can be batched.
- OBB fields in `EvlBackboneOutput` still not batch-collated.

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests/lightning/test_vin_batch_collate.py`

## Follow-up fixes and integration checks (2026-01-03)
- Fixed `pos_grid_from_pts_world` batch-center broadcasting: now unsqueezes voxel center to `(B,1,3)` before PoseTW transforms, preventing the erroneous `(B,B,3)` center_rig shape when `center_vox` was `(B,3)`.
- Manual CPU integration run (num_workers=1, batch_size=2) with a synthetic offline cache + stubbed depth decoder produced two batched forward passes through `VinModelV2`.
  - Batch 0: `candidate_poses` (2,3,12), `snippet points` (2,30,4), `traj` (2,5,12), `backbone` occ_pr (2,1,2,2,2), `pred logits` (2,3,14).
  - Batch 1: `candidate_poses` (2,4,12), `snippet points` (2,56,4), `traj` (2,7,12), `backbone` occ_pr (2,1,2,2,2), `pred logits` (2,4,14).
  - NaNs appear in padded RRI entries as expected for variable candidate counts.

## Aux regression mask fix (2026-01-03)
- Fixed aux regression loss indexing in `oracle_rri/oracle_rri/lightning/lit_module.py`: use flattened `rri_valid` and `pred_rri_proxy_valid` instead of indexing the 2D `rri` with a flattened mask, which caused `IndexError` when batching (e.g. mask shape 240 vs rri shape 4x60).

## VinSnippet cache (2026-01-03)
- Added `oracle_rri/oracle_rri/data/vin_snippet_cache.py` with configs, writer, dataset, and metadata for a minimal VIN snippet cache (collapsed semidense points + trajectory).
- Integrated optional `vin_snippet_cache` into `OracleRriCacheDatasetConfig`; when set and `return_format="vin_batch"`, the dataset loads `VinSnippetView` from the cache and skips full EFM snippet loading.
- Updated `oracle_rri/oracle_rri/data/README.md` with usage + TOML snippet for enabling the cache.

Open:
- Need a real-data integration run to validate the writer + cache read path on the actual offline cache.

## VinSnippet cache tests (2026-01-03)
- Unit test: `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache.py`.
- Real-data integration test: `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache_real_data.py` (passes on local `.data/oracle_rri_cache`).

## CLI + offline_only wiring (2026-01-04)
- Added new console script `nbv-cache-vin-snippets` (see `oracle_rri/pyproject.toml`) to build the VIN snippet cache directly from an experiment TOML (e.g. `.configs/offline_only.toml`).
  - Example: `uv run nbv-cache-vin-snippets --config-path .configs/offline_only.toml --split train --overwrite --max-samples 100`.
- Updated `.configs/offline_only.toml` to configure `datamodule_config.*_cache.vin_snippet_cache.cache_dir = "vin_snippet_cache"`.
- Made `vin_snippet_cache` robust: if the cache index is missing, `OracleRriCacheDataset` warns once and falls back to EFM snippet loading.

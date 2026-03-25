---
id: 2026-03-24_test_failure_investigation
date: 2026-03-24
title: "Test Failure Investigation"
status: done
topics: [tests, migration, scaffolding]
confidence: medium
canonical_updates_needed: []
source_legacy_path: ".codex/test_failure_investigation_2026-03-24.md"
---

> Migrated from a leftover `.codex` note during the 2026-03-25 scaffold simplification pass.

# Test Failure Investigation (2026-03-24)

## Scope
Investigated all current failures in `oracle_rri/tests` against current UUT contracts.

## Executed Commands
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests -q -s --continue-on-collection-errors`
- Per-file first-cause sweep with `--maxfail=1` for every failing/erroring file.

## Isolation Fix Implemented
- Added global autouse fixture:
  - `oracle_rri/tests/conftest.py`
  - Snapshots/restores `PathConfig` singleton around each test.
- Added regression:
  - `oracle_rri/tests/test_pathconfig_isolation_regression.py`
  - Verifies path mutation in one test does not leak into the next.

## Status Change After Isolation Fix
- Before: `71 failed, 16 errors, 169 passed, 13 skipped, 1 xfailed`
- After: `55 failed, 10 errors, 191 passed, 15 skipped, 1 xfailed`

## Remaining Failure Clusters (31 files)

### A) Missing/removed symbols or modules (legacy tests)
- `app/panels/test_wandb_panel.py`
  - Missing: `_flatten_mapping`, `_extract_run_steps`, `_filter_runs` in `configs.wandb_config`; `_summarize_metric`, `_summarize_gap` in panel module.
- `pose_generation/test_pose_generation_revised.py`
  - Missing module: `reference_power_spherical_distributions`.
- `rendering/test_rendering_plotting_helpers.py`
  - Missing symbol: `hit_ratio_bar`.
- `test_mesh_cache.py`
  - Missing symbol: `mesh_from_snippet`.
- `test_mesh_cropping.py`
  - Missing symbol: `crop_mesh_with_bounds`.
- `test_performance_mode.py`
  - Missing module export: `oracle_rri.utils.performance`.

### B) API/config schema drift in tests
- `data_handling/test_dataset.py`
  - Uses removed kwarg `verbose` on `AseEfmDatasetConfig`.
  - Patches removed symbol `load_atek_wds_dataset_as_efm` (now loader call path changed).
- `data_handling/test_real_data_integration.py`
  - Uses removed kwarg `verbose` on `AseEfmDatasetConfig`.
- `rendering/test_candidate_renderer_integration.py`
  - Uses removed kwarg `verbose` on `AseEfmDatasetConfig`.
- `test_pose_generation.py`
  - Uses removed kwarg `verbose` on `AseEfmDatasetConfig`.
- `test_efm_dataset.py`
  - `test_batching_supported` uses removed kwarg `verbose`.
- `rendering/test_candidate_renderer_cpu_backend.py`
  - Uses removed kwarg `verbose` on `Pytorch3DDepthRendererConfig`.
- `rendering/test_pytorch3d_renderer.py`
  - Uses removed kwarg `verbose` on `Pytorch3DDepthRendererConfig`.
- `rri_metrics/test_oracle_rri_chunking.py`
  - Uses removed kwarg `candidate_chunk_size` on `OracleRRIConfig`.
- `data/test_vin_snippet_cache_datamodule_equivalence.py`
  - Uses removed kwarg `map_location` on `OracleRriCacheDatasetConfig`.
- `lightning/test_vin_batch_collate.py`
  - `VinSnippetView` constructor now requires `lengths`.

### C) Behavioral/assertion drift vs current UUT
- `pose_generation/test_align_to_gravity.py`
  - Assertions expect previous orientation/elevation behavior.
- `rendering/test_depth_backprojection_conventions.py`
  - Expected sign convention no longer matches current backprojection convention.
- `rendering/test_pytorch3d_depth_renderer.py`
  - Renderer now returns 3-tuple; test expects 2 outputs.
- `vin/test_learnable_fourier_features.py`
  - Expects old gamma-squared init; current UUT uses gamma scaling.
- `lightning/test_wandb_logging.py`
  - Logger interaction assertion (`assert logged`) no longer matches contract.

### D) Test-internal drift (paths/imports/monkeypatch targets)
- `data/test_offline_cache.py`
  - Imports `tests.vin.utils` (old root-tests path), now missing.
- `data/test_offline_cache_pruning.py`
  - Monkeypatch helper signature stale (`decode_depths(..., device=...)` now passes `device`).
- `data/test_offline_cache_split.py`
  - Assumes wrapper dataset exposes `.config`; `_ShuffleCandidatesMapDataset` does not.
- `pose_generation/test_candidate_generation_mesh_access.py`
  - Monkeypatch target `candidate_generation.mesh_from_snippet` no longer exists.
- `lightning/test_reduce_lr_on_plateau_config.py`
  - Broken migrated filesystem path (`.../oracle_rri/oracle_rri/oracle_rri/lightning/optimizers.py`).

### E) Legacy data_handling tests for removed/changed data contracts
- `data_handling/test_downloader_cli_list_snippet_totals.py`
  - Output expectation mismatch with current downloader CLI behavior.
- `data_handling/test_mesh_cache.py`
  - `MeshProcessSpec` signature changed; stale `snippet_id` argument.
- `data_handling/test_metadata.py`
  - `SceneMetadata` signature changed; stale `snippet_count` argument.

## Recommended Next Fix Order
1. Resolve collection errors and removed-symbol imports (Cluster A).
2. Bulk update stale config kwargs and constructor signatures (Cluster B).
3. Fix test-internal migration breakages (Cluster D).
4. Rebaseline behavioral assertions against canonical UUT (Cluster C).
5. Decide keep/rewrite/drop policy for legacy `data_handling/*` tests (Cluster E), then apply consistently.

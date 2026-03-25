---
id: 2026-03-24_test_run_overview_2026-03-24
date: 2026-03-24
title: "Test Run Overview 2026 03 24"
status: legacy-imported
topics: [test, run, overview, 2026, 03]
source_legacy_path: ".codex/test_run_overview_2026-03-24.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Test Run Overview (2026-03-24)

## Commands
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests -q -s`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests -q -s --continue-on-collection-errors`

## Environment
- Python: 3.11.14
- Torch: 2.4.1+cu121
- Pytest: 9.0.1

## Results
- Default run (no continue): `exit=2`
  - Collected 262 items
  - 5 collection errors
  - 3 skipped at collection phase
- Continue-on-collection-errors run: `exit=1`
  - `169 passed`
  - `71 failed`
  - `16 errors`
  - `13 skipped`
  - `1 xfailed`

## Collection Blockers (UUT/API mismatches)
1. `oracle_rri.pose_generation.reference_power_spherical_distributions` missing
2. `oracle_rri.rendering.plotting.hit_ratio_bar` missing export
3. `oracle_rri.data.mesh_cache.mesh_from_snippet` missing export
4. `oracle_rri.data.efm_dataset.crop_mesh_with_bounds` missing export
5. `oracle_rri.utils.performance` missing export/module

## UUT Impact (from failing+erroring tests import mapping)
- `oracle_rri.data`: 62
- `oracle_rri.configs`: 37
- `oracle_rri.utils`: 35
- `oracle_rri.pose_generation`: 24
- `oracle_rri.rendering`: 16
- `oracle_rri.lightning`: 9
- `oracle_rri.vin`: 9
- `oracle_rri.app`: 7
- `oracle_rri.rri_metrics`: 6
- `oracle_rri.pipelines`: 3

## Dominant Failure Signatures
- Pydantic config validation drift (`extra_forbidden`, changed/removed config fields).
- Dataset/config path assumptions (`AseEfmDatasetConfig.tar_urls` validation failures).
- Pose-generation API drift (`last_pose` parameter no longer accepted in generator/context).
- Rendering API drift (constructor args and render return shape expectations changed).
- Removed or renamed helper methods (`collapse_points_np`, plotting/export helpers).

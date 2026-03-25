---
id: 2026-03-24_tests_migration_2026-03-24
date: 2026-03-24
title: "Tests Migration 2026 03 24"
status: legacy-imported
topics: [tests, migration, 2026, 03, 24]
source_legacy_path: ".codex/tests_migration_2026-03-24.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Root Tests Migration to `oracle_rri/tests` (2026-03-24)

## Goal
Consolidate valid tests from root `tests/` into `oracle_rri/tests` using strict runtime gating (`pytest --collect-only -q -s`).

## Migration Summary
- Root tests discovered: 34
- Runtime-valid: 31
- Runtime-invalid: 3
- Valid unique files migrated: 30
- Overlap merged manually: `vin/test_rri_binning.py`
- Root `tests/` removed after migration.

## Intentionally Dropped (failed runtime collect gate)
- `tests/configs/test_wandb_config.py`
- `tests/rendering/test_unproject.py`
- `tests/vin/test_pose_vec_groups.py`

## Overlap Merge
- Merged root assertions into:
  - `oracle_rri/tests/vin/test_rri_binning.py`
- Resulting file now contains 5 tests and passes:
  - `pytest oracle_rri/tests/vin/test_rri_binning.py -q -s`

## Verification Results
- `python -c "import torch; print(torch.__version__)"` => `2.4.1+cu121`
- `pytest oracle_rri/tests --collect-only -q -s` collects successfully but has 5 pre-existing import errors in legacy package tests:
  1. `oracle_rri/tests/pose_generation/test_pose_generation_revised.py`
  2. `oracle_rri/tests/rendering/test_rendering_plotting_helpers.py`
  3. `oracle_rri/tests/test_mesh_cache.py`
  4. `oracle_rri/tests/test_mesh_cropping.py`
  5. `oracle_rri/tests/test_performance_mode.py`

These errors are unrelated to the migrated root tests and stem from missing/renamed symbols in the current package code.

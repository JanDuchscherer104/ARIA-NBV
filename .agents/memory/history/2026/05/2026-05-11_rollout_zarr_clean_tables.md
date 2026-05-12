---
id: 2026-05-11_rollout_zarr_clean_tables
date: 2026-05-11
title: "Rollout Zarr Clean Tables"
status: done
topics: [data-handling, rollouts, zarr, simplification]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/data_handling/_rollout_zarr_store.py
  - aria_nbv/aria_nbv/data_handling/_target_selection.py
  - aria_nbv/aria_nbv/pose_generation/target_counterfactuals.py
  - aria_nbv/aria_nbv/app/panels/candidates.py
  - aria_nbv/aria_nbv/data_handling/README.md
  - aria_nbv/tests/data_handling/test_rollout_zarr_store.py
  - aria_nbv/tests/pose_generation/test_counterfactuals.py
---

## Task

Implemented the requested ruthless simplification pass for multi-step rollout data handling.

## Changes

- Bumped `rollouts.zarr` to schema `0.2-clean-tables`.
- Removed the shard-local `splits/` group and made validation reject shards that mix split ids.
- Stopped mirroring rollout fields into `lineage/`; `lineage/` now owns rollout-row id plus source/config/protocol hash fields only.
- Kept target-selection and GT-match details in `targets/`, with rollouts linking by `target_row_id`.
- Replaced prefix-routed table flattening with explicit table schemas and one row-to-array conversion path.
- Preserved public writer/reader functions and the MessagePack rollout trace smoke path.

## Verification

- `cd aria_nbv && uv run pytest tests/data_handling/test_rollout_zarr_store.py tests/data_handling/test_target_selection.py -q`
- `cd aria_nbv && uv run pytest tests/rerun_inspector tests/app/panels -q`
- `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/data_handling/_rollout_zarr_store.py aria_nbv/data_handling/_target_selection.py aria_nbv/pose_generation/target_counterfactuals.py aria_nbv/app/panels/candidates.py tests/data_handling/test_rollout_zarr_store.py tests/pose_generation/test_counterfactuals.py`

## State Impact

Old `rollouts.zarr` shards are intentionally not backward-compatible with the cleaned schema and should fail validation. The current source of truth for the cleaned table layout is `aria_nbv/aria_nbv/data_handling/README.md`.

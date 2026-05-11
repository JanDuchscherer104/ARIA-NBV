---
id: 2026-05-11_rollout_dataset_remediation_inspector_alignment
date: 2026-05-11
title: "Rollout Dataset Remediation And Inspector Alignment"
status: done
topics: [rollouts, target-rri, zarr, streamlit, rerun]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/pose_generation/target_counterfactuals.py
  - aria_nbv/aria_nbv/data_handling/_rollout_dataset_writer.py
  - aria_nbv/aria_nbv/data_handling/_rollout_zarr_store.py
  - aria_nbv/aria_nbv/app/panels/candidates.py
  - aria_nbv/aria_nbv/app/panels/offline_dataset.py
  - aria_nbv/aria_nbv/app/rerun_launch.py
---

## Task

Remediated the fresh rollout-generation path before treating it as thesis evidence. The pass focused on explicit target/scene RRI semantics, oriented GT-OBB target crops, stronger lineage and post-write validation, fail-fast generation, and Streamlit/Rerun inspector alignment.

## Method

Target-RRI scoring now resolves the matched GT OBB in world coordinates, crops target mesh faces with the `gt_obb_oriented_any_vertex_v1` policy, and computes target and diagnostic scene RRI from the same rendered/backprojected candidate point-cloud batch. The rollout writer now hashes actual source rows and validates non-synthetic rollout Zarr stores immediately after writing. Unexpected rollout/scorer errors now fail fast; only expected target-label invalidity is counted as a skip.

The duplicate candidate-strategy enum was removed in favor of `ViewDirectionMode` plus a stable strategy-id mapping. Streamlit gained a lightweight `rollouts.zarr` summary/launcher tab on the candidate page and an offline-sample Rerun launch path on the offline dataset page. Rerun remains the rich inspector.

## Verification

- `cd aria_nbv && uv run pytest tests/pose_generation tests/data_handling/test_rollout_zarr_store.py tests/data_handling/test_rollout_dataset_writer.py tests/rerun_inspector tests/app/test_rerun_launch.py tests/app/panels -q`
- `cd aria_nbv && uv run pytest tests/data_handling tests/rri_metrics -q`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`
- `cd aria_nbv && uv run ruff check <touched Python files>`

## Canonical State Impact

No canonical state file update is required. The current thesis direction already treats V1 target-specific rollout data as the next evidence path and keeps VIN strict-v7 unchanged.

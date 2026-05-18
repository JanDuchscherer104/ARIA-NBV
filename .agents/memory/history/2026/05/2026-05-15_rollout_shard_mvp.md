---
id: 2026-05-15_rollout_shard_mvp
date: 2026-05-15
title: "Rollout Shard MVP"
status: done
topics: [rollouts, lrz, zarr, data-handling, lineage]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/data_handling/_offline_dataset.py
  - aria_nbv/aria_nbv/rollouts/
  - aria_nbv/tests/rollouts/
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
  - scripts/templates/lrz/rollout_generation_dry_run.sbatch
  - .agents/todos.toml
  - .agents/resolved.toml
---

## Task

Implemented the rollout shard MVP selected on 2026-05-15: source-row shard manifests,
strict resumable temp-to-final shard writes, and complete VIN source-shard lineage
inside standalone rollout stores.

## Method

- Added `VinOfflineSample.source_shard_id` and `source_shard_row` populated from
  `VinOfflineIndexRecord`; `VinOracleBatch` remains unchanged.
- Added rollout JSONL shard manifest records, deterministic shard planning,
  `nbv-plan-rollout-shards`, and shard-aware `nbv-build-rollouts` options.
- Added strict shard execution: skip only validated final shards with matching
  `_SUCCESS.json` and `_owner.json`; fail stale temp or partial final paths.
- Strengthened rollout Zarr validation so `sources/source_shard_id` must resolve
  to non-empty VIN source shard ids and `sources/source_shard_row` must be
  non-negative.
- Updated the LRZ rollout dry-run template to call the real shard-aware CLI.
- Resolved `todo-076` and `todo-077` in agents DB; `todo-078` remains active for
  persisted chunked `Q_H` training views.

## Verification

- `cd aria_nbv && uv run ruff format ...`
- `cd aria_nbv && uv run ruff check ...`
- `bash -n scripts/templates/lrz/rollout_generation_dry_run.sbatch`
- `cd aria_nbv && uv run pytest tests/rollouts tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`
- `cd aria_nbv && uv run nbv-plan-rollout-shards --config-path ../.configs/build_rollouts_v1_smoke.toml --output-manifest /tmp/rollout_shards_smoke.jsonl --rows-per-shard 1 --dry-run`

## Canonical Impact

No separate canonical-state edit is required. The implemented package contracts,
LRZ template, tests, resolved agents-DB records, and this debrief now capture the
current state. The remaining throughput blocker is the materialized chunked
`Q_H` training view tracked by `todo-078`.

---
id: 2026-05-07_rollout_zarr_review_blockers_fixed
date: 2026-05-07
title: "Rollout Zarr Review Blockers Fixed"
status: done
topics: [rollouts, zarr, q-h, data, agents-db]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/data_handling/_rollout_zarr_store.py
  - aria_nbv/tests/data_handling/test_rollout_zarr_store.py
  - .agents/issues.toml
---

## Task

Fixed the critical review blockers in the standalone `rollouts.zarr` tracer
bullet before `Q_H` training consumes rollout replay arrays.

## Method

Hardened the writer and validator so trainable target-RRI labels require
explicit target-RRI metric vectors, invalid candidates keep target labels as
`NaN`, multi-target row ids propagate into `q_h/target_row_id`, root-relative
candidate poses are true `PoseTW` transforms, and non-synthetic stores reject
synthetic or missing lineage. Added regression tests for non-oracle score
fallbacks, invalid candidate labels, multi-target identity, relative pose math,
per-rollout lineage, and non-synthetic lineage rejection.

Amended `issue-019` so the active stochastic rollout debt reflects the current
state: temperature-softmax exists, while Gumbel-Top-k, branch schedules, late
exploration, and scale evidence remain open.

## Verification

- `cd aria_nbv && .venv/bin/ruff format aria_nbv/data_handling/_rollout_zarr_store.py tests/data_handling/test_rollout_zarr_store.py`
- `cd aria_nbv && .venv/bin/ruff check aria_nbv/data_handling/_rollout_zarr_store.py tests/data_handling/test_rollout_zarr_store.py aria_nbv/pose_generation/rollout_trace.py`
- `cd aria_nbv && uv run pytest tests/data_handling/test_rollout_zarr_store.py tests/pose_generation/test_counterfactuals.py`
- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`

## Canonical State Impact

No durable thesis direction changed. The fixed behavior implements the existing
rollout/Q_H invalidity and storage contract; remaining large-scale storage work
stays tracked by `issue-018`.

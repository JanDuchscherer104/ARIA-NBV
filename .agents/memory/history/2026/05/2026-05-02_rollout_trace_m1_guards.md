---
id: 2026-05-02_rollout_trace_m1_guards
date: 2026-05-02
title: "Rollout Trace And M1 Guard Tests"
status: done
topics: [rollouts, geometry, offline-store, agents-db, thesis]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/pose_generation/rollout_trace.py
  - aria_nbv/aria_nbv/pose_generation/__init__.py
  - aria_nbv/aria_nbv/lightning/aria_nbv_experiment.py
  - aria_nbv/aria_nbv/rerun_inspector/_sample.py
  - aria_nbv/aria_nbv/rerun_inspector/_metadata.py
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/tests/pose_generation/test_counterfactuals.py
  - aria_nbv/tests/vin/test_vin_utils.py
  - aria_nbv/pyproject.toml
  - scripts/agents_db.py
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/resolved.toml
artifacts:
  - .artifacts/rollouts/synthetic_rollout_traces.msgpack
---

## Task

Proceed with the selected backlog items: M1 geometry/candidate-order guards,
M1 data/cache/oracle contract stabilization, offline_only smoke validation, and
the first RolloutTrace schema/writer.

## Method

Added a compact rollout trace module that converts `CounterfactualRolloutResult`
into lineage-bearing `RolloutTrace` records, writes/reads MessagePack payloads,
and exposes `nbv-rollout-trace-smoke` for a synthetic one-scene greedy/random
trace. Added guard tests for candidate serialization order, display-only CW90
behavior, full-shell candidate indices, and trace score/mask alignment.

Fixed summary-mode config defaults so online sources stay unbatched while
offline map-style sources still use batch size 1. Patched the untracked Rerun
inspector surface to import `VinOfflineSample`/dataset types through the public
`aria_nbv.data_handling` package root so the public API contract passes.

## Findings

`uv run nbv-summary --config-path offline_only.toml` now reaches strict
offline-store validation, but the local `.data/offline_cache/vin_offline`
manifest is version 5 while the current reader expects version 6. The
offline_only smoke todo remains active and the agents DB issue/todo now record
that exact blocker.

The agents-db resolver could not write resolved records while legacy resolved
items lacked newer required fields. `scripts/agents_db.py` now tolerates missing
legacy fields when rewriting `.agents/resolved.toml`; validation still enforces
required fields for active records.

## Verification

- `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py -q`
- `cd aria_nbv && uv run pytest tests/vin/test_vin_utils.py -q`
- `cd aria_nbv && uv run nbv-rollout-trace-smoke --output-path ../.artifacts/rollouts/synthetic_rollout_traces.msgpack --horizon 2 --num-samples 8 --seed 0`
- `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py tests/rendering/test_depth_backprojection_conventions.py tests/lightning/test_vin_batch_collate.py tests/vin/test_vin_utils.py tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py -q`
- `cd aria_nbv && uv run pytest tests/rri_metrics -q`
- `make context-contracts`
- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`

## Canonical Impact

No durable state files need updates yet. The current truth remains that M1 is
not fully frozen until the local offline store is rebuilt or migrated to the
current immutable-store version and oracle throughput is measured.

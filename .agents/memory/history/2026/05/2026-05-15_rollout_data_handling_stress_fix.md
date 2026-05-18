---
id: 2026-05-15_rollout_data_handling_stress_fix
date: 2026-05-15
title: "Rollout Data Handling Stress Fix"
status: done
topics: [data-handling, rollouts, counterfactuals, validation]
confidence: high
canonical_updates_needed: []
---

## Task

Stress-tested the current data-handling and counterfactual rollout generation
path after the rollout shard MVP. Scope covered immutable VIN source rows,
rollout shard manifests/resume, rollout Zarr lineage, and finite-candidate
counterfactual selection.

## Findings And Fixes

- Real smoke shard generation initially failed when the source config exposed
  `split = "all"` but the manifest owned a split-local `train` row. Fixed
  rollout split lineage hashing to use the concrete selected row split when
  source rows are split-local.
- Tightened shard resume checks so `_SUCCESS.json` must name the current
  `_owner.json` hash, the rollout manifest hash must match both sidecars, and
  the embedded store manifest shard entry must match the requested manifest.
- Rejected conflicting source lineage for repeated rollout `source_row_id`
  values before Zarr materialization.
- Canonicalized unpadded CLI shard ids such as `shard-0` to `shard-000000`.
- Hardened counterfactual selection so non-finite evaluator scores are not
  selected and temperature-softmax probabilities remain finite.
- Added docstrings clarifying the handoff from `VinOfflineSample` roots to
  `pose_generation` counterfactual trees and standalone rollout stores.

## Verification

- `cd aria_nbv && uv run ruff format ...`
- `cd aria_nbv && uv run ruff check ...`
- `cd aria_nbv && uv run pytest tests/data_handling tests/pose_generation/test_counterfactuals.py tests/rollouts`
- `cd aria_nbv && uv run nbv-plan-rollout-shards --config-path ../.configs/build_rollouts_v1_smoke.toml --output-manifest /tmp/rollout_shards_stress.jsonl --rows-per-shard 1`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --shard-manifest /tmp/rollout_shards_stress.jsonl --shard-id shard-000000 --output-tmp /tmp/rollout_shard_stress_tmp --output-final /tmp/rollout_shard_stress_final`
- `cd aria_nbv && uv run nbv-rollouts-info --store /tmp/rollout_shard_stress_final --validate --json`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --shard-manifest /tmp/rollout_shards_stress.jsonl --shard-id shard-0 --output-tmp /tmp/rollout_shard_stress_tmp_strict --output-final /tmp/rollout_shard_stress_final`
- `git diff --check -- <touched paths>`

## Canonical State Impact

No canonical state update is required. The fixed behavior matches the existing
thesis direction: immutable VIN stores remain the one-step substrate, rollout
replay is standalone, and LRZ generation must be shard-local and resumable.

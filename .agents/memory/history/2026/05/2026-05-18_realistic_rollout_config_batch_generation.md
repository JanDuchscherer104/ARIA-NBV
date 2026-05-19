---
id: 2026-05-18_realistic_rollout_config_batch_generation
date: 2026-05-18
title: "Realistic Rollout Config Batch Generation"
status: done
topics: [rollouts, candidate-generation, dataset-cache]
confidence: high
canonical_updates_needed: []
files_touched:
  - .configs/build_rollouts_v1_realistic.toml
artifacts:
  - .data/offline_cache/rollouts_v1_realistic.zarr
---

## Task
Tighten the realistic V1 candidate-offset configuration, generate a fresh local rollout batch, and validate the resulting rollout store.

## Method
Applied the ARIA agent-behavior, dataset-cache, and counterfactual-rollout guidance. Patched `.configs/build_rollouts_v1_realistic.toml` to use the train split, a tighter local motion envelope, higher oversampling, and a rebalance toward forward-local candidates. Ran the existing CLI only:

- `uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_realistic.toml --dry-run`
- `uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_realistic.toml`
- `uv run nbv-rollouts-info --store ../.data/offline_cache/rollouts_v1_realistic.zarr --validate`

## Findings
The first generated attempt exposed that `source.split = "all"` violates the current rollout-store single-split validation rule, so the config now uses `split = "train"`. The initially stricter motion envelope made `local_refinement` and `revisit_backtrack` dead; the final config keeps a conservative envelope while using `forward_rig` for local refinement and bounded revisit/backtrack.

Final generated store:

- path: `.data/offline_cache/rollouts_v1_realistic.zarr`
- schema: `0.7-root-gain-target-crops`
- rows: 24 rollouts, 40 steps, 2400 candidates
- valid candidates: 536 / 2400 (0.223)
- valid per step: min 6, median 9.5, mean 13.4, max 37
- selected path length: min 0.041 m, median 0.286 m, mean 0.386 m, max 0.807 m
- invalid candidates are dominated by `CLEARANCE_TOO_SMALL` with secondary `PATH_SEGMENT_COLLISION`

Residual issue: candidate validity remains low on geometry-tight steps; this is now mostly clearance/path limited rather than a dead orientation component.

## Verification
Validation passed:

- `uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_realistic.toml --dry-run`
- `uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_realistic.toml`
- `uv run nbv-rollouts-info --store ../.data/offline_cache/rollouts_v1_realistic.zarr --validate`

No Python tests were run because the change was a TOML config and generated ignored rollout artifact, not package code.

## Canonical State Impact
None.

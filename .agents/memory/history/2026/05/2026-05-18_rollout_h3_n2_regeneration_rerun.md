---
id: 2026-05-18_rollout_h3_n2_regeneration_rerun
date: 2026-05-18
title: "Rollout H3 N2 Regeneration And Rerun Inspection"
status: done
topics: [rollouts, rerun, data-handling, selected-depth]
confidence: high
canonical_updates_needed: []
artifacts:
  - .data/offline_cache/rollouts_v1_smoke.zarr
  - .artifacts/rerun/rollouts_v1_smoke_h3_n2_idx000_random.rrd
  - .artifacts/rerun/rollouts_v1_smoke_h3_n2_idx001_greedy.rrd
  - .artifacts/rerun/rollouts_v1_smoke_h3_n2_idx004_softmax.rrd
---

## Task

Deleted old local rollout-derived cache and Rerun artifacts, regenerated a small CUDA-backed H=3 rollout smoke store, and inspected saved Rerun recordings until the selected-depth and indexing conventions were confirmed.

## Method

Old rollout artifacts were removed from `.data/offline_cache` and `.artifacts/rerun` while preserving the VIN offline source stores and non-rollout Rerun files. A scratch config at `/tmp/aria-nbv-rollouts-h3-n2-cuda.toml` was derived from `.configs/build_rollouts_v1_smoke.toml` with `max_samples = 2`, `source.limit = 2`, CUDA rendering/scoring, `target_selector.k = 2`, and all recipes set to `horizon = 3`.

The first generation attempt with `source.split = "all"` was rejected by post-write validation because a rollout shard must contain exactly one split. The invalid store was deleted and the scratch config was changed to `source.split = "train"`, after which generation completed.

## Outputs

The regenerated store is `.data/offline_cache/rollouts_v1_smoke.zarr` with schema `0.6-rollout-core`. `nbv-rollouts-info --validate --json` reported no validation errors and these counts:

- sources: 2
- targets: 3
- rollouts: 18
- steps: 54
- candidates: 270
- selected depths: 54
- Q_H states: 54

Selected-depth inspection confirmed `depth_m` is `float16` with shape `(54, 240, 240)` and chunks `(16, 240, 240)`, `valid_mask` is `bool`, selected-depth `step_row_id` and `candidate_row_id` match factual `steps/`, and all selected candidates are feasible with finite target RRI.

Three Rerun recordings were saved for a random-valid, oracle-greedy, and temperature-softmax rollout:

- `.artifacts/rerun/rollouts_v1_smoke_h3_n2_idx000_random.rrd`
- `.artifacts/rerun/rollouts_v1_smoke_h3_n2_idx001_greedy.rrd`
- `.artifacts/rerun/rollouts_v1_smoke_h3_n2_idx004_softmax.rrd`

`rerun rrd print` inspection confirmed each recording has exactly three scoped selected depth entities under `rollout_*/chain_000000/step_*/selected/candidate_shell_*/camera/depth`, with no old unscoped `/world/rollout/step/*` or `/world/rollout/selected_path` entities. The recordings still include `/world/candidates`, `/world/efm/voxels`, and `/world/gt/obbs`; the active blueprint hides those by default and also hides rollout step `valid` and `invalid` groups by default.

## Verification

- `cd aria_nbv && uv run nbv-build-rollouts --config-path /tmp/aria-nbv-rollouts-h3-n2-cuda.toml --dry-run`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path /tmp/aria-nbv-rollouts-h3-n2-cuda.toml`
- `cd aria_nbv && uv run nbv-rollouts-info --store ../.data/offline_cache/rollouts_v1_smoke.zarr --validate --json`
- `cd aria_nbv && uv run nbv-rerun-inspect ... --rollout-index 0 --save ../.artifacts/rerun/rollouts_v1_smoke_h3_n2_idx000_random.rrd`
- `cd aria_nbv && uv run nbv-rerun-inspect ... --rollout-index 1 --save ../.artifacts/rerun/rollouts_v1_smoke_h3_n2_idx001_greedy.rrd`
- `cd aria_nbv && uv run nbv-rerun-inspect ... --rollout-index 4 --save ../.artifacts/rerun/rollouts_v1_smoke_h3_n2_idx004_softmax.rrd`
- `cd aria_nbv && uv run rerun rrd print -v ../.artifacts/rerun/rollouts_v1_smoke_h3_n2_idx000_random.rrd`
- `cd aria_nbv && uv run pytest tests/rerun_inspector/test_loggers.py::test_default_blueprint_hides_heavy_context_and_requested_world_subtrees tests/rerun_inspector/test_rollout_zarr_logger.py::test_rollout_zarr_logger_logs_multistep_candidate_layers -q`
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector/_loggers.py tests/rerun_inspector/test_loggers.py tests/rerun_inspector/test_rollout_zarr_logger.py`

## Canonical State Impact

No canonical state update is needed. The mixed-split rejection is expected behavior from the current validation contract, and the generated artifacts are local smoke data rather than thesis-scale source data.

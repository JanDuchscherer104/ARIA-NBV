---
id: 2026-05-12_rollout_smoke_generation_and_rerun_inspection_fixes
date: 2026-05-12
title: "Rollout smoke generation and Rerun inspection fixes"
status: done
topics: [rollouts, rerun, target-rri, zarr]
confidence: high
canonical_updates_needed: []
---

## Task
Fix the local rollout smoke generation path so `nbv-build-rollouts` produces inspectable V1 target-conditioned rollout samples and a valid Rerun `.rrd`.

## Method
Reproduced the empty-record crash, inspected the first VIN offline source row, and traced the failure to zero selected V1 targets: the local strict-v7 store has no persisted compact OBB blocks, and predicted backbone OBBs have no 2D projected boxes. Patched the active path to use live raw-snippet GT OBBs for GT-EVAL when compact GT OBBs are not materialized, allow supported 3D predicted OBB targets without 2D projection by default, persist actor-visible target OBB fields in `rollouts.zarr`, and make Rerun target highlighting use matched GT target ids.

## Findings
- `.configs/build_rollouts_v1_smoke.toml` is now an actual CPU smoke config: 8 mixed candidates, coarser depth/backprojection, and retained one/two-step recipes.
- `aria_nbv/aria_nbv/data_handling/_offline_dataset.py` can attach live raw-snippet GT OBBs for label/evaluation when the immutable store predates compact OBB persistence.
- `aria_nbv/aria_nbv/data_handling/_target_selection.py` no longer hard-masks otherwise supported backbone 3D OBBs solely because their 2D projected boxes are unavailable.
- `aria_nbv/aria_nbv/rollouts/{dataset_writer.py,trace.py,zarr_store.py}` now persists target OBB source, semantic/instance ids, confidence, center, extents, and world/object pose fields in `targets/`.
- `aria_nbv/aria_nbv/rerun_inspector/{_rollout_zarr.py,_loggers.py}` now prefers matched GT target ids for OBB highlighting, understands `sem=...` / `inst=...` rollout target tokens, and logs only the latest valid OBB slice to avoid duplicate target boxes.

## Verification
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml` passed, writing 6 rollout rows, 10 step rows, and 80 candidate rows to `.data/offline_cache/rollouts_v1_smoke.zarr`.
- `RolloutZarrStoreReader(...).validate()` passed with no errors; `q_h_view()` reported 10 states, 55 valid/train actions, and finite target-RRI labels.
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline.toml --rollout-store ../.data/offline_cache/rollouts_v1_smoke.zarr --rollout-index 0 --rollout-context required --save ../.artifacts/rerun/rollout_v1_smoke.rrd` passed and wrote a 4.7 MiB `.rrd`.
- `cd aria_nbv && uv run rerun rrd print ../.artifacts/rerun/rollout_v1_smoke.rrd` showed GT OBB, rollout metadata, selected rollout camera, selected center, and selected path chunks.
- `cd aria_nbv && uv run pytest tests/data_handling/test_target_selection.py tests/rollouts/test_zarr_store.py tests/rerun_inspector/test_loggers.py tests/rerun_inspector/test_rollout_zarr_logger.py -q` passed: 46 passed, 1 third-party deprecation warning.
- `cd aria_nbv && uv run ruff check ...` and `cd aria_nbv && uv run ruff check --select F401,TCH ...` passed for touched files.

## Canonical State Impact
No canonical state update required. This records implementation and validation detail for the rollout smoke/debug path, not a new thesis direction.

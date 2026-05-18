---
id: 2026-05-18_rerun_selected_depth_rollout_inspection
date: 2026-05-18
title: "Rerun Selected-Depth Rollout Inspection"
status: done
topics: [rerun, rollouts, selected-depth, inspection]
confidence: high
canonical_updates_needed: []
files_touched:
  - .configs/rerun_offline.toml
  - aria_nbv/aria_nbv/rerun_inspector/_config.py
  - aria_nbv/aria_nbv/rerun_inspector/_rollout_zarr.py
  - aria_nbv/tests/rerun_inspector/test_rollout_zarr_logger.py
  - aria_nbv/tests/test_config_field_constraints.py
artifacts:
  - .artifacts/rerun/rollout_selected_depth.rrd
---

## Task

Implemented native Rerun inspection for `rollouts.zarr/selected_depth` so
materialized selected counterfactual poses can display their metric depth image
as a child of the selected camera/frustum entity.

## Method

Added a `rollout_depths` inspector config block, lazily joined rollout
`steps/step_row_id` to `selected_depth/step_row_id`, checked selected candidate
identity, converted invalid depth pixels to `NaN`, and logged
`Transform3D + Pinhole + DepthImage` under the selected candidate camera path.
Selected cameras with depth now use persisted focal length, principal point,
and `[W,H]` resolution instead of the generic 90-degree fallback.

The local `.rrd` smoke on the old `rollouts_v1_smoke.zarr` store exposed that
the inspector's Q_H metadata path still called `q_h_view()`, which eagerly tried
to read the new `selected_depth` group. Reworked that metadata helper to derive
the small fields it needs directly from `steps/` and `candidates/`, preserving
old-store warning behavior and avoiding eager depth-table reads.

## Verification

- `cd aria_nbv && uv run pytest tests/rerun_inspector/test_rollout_zarr_logger.py`
- `cd aria_nbv && uv run pytest tests/rerun_inspector`
- `cd aria_nbv && uv run pytest tests/test_config_field_constraints.py`
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector tests/rerun_inspector tests/test_config_field_constraints.py`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline.toml --rollout-store ../.data/offline_cache/rollouts_v1_smoke.zarr --rollout-index 0 --save ../.artifacts/rerun/rollout_selected_depth.rrd`

The smoke artifact was written successfully on 2026-05-18. The local smoke
store is old and has no `selected_depth` group, so it exercises compatibility
and metadata warnings rather than visible selected-depth rasters.

## Canonical State Impact

No canonical state update is required. The code/config/test surfaces now encode
the selected-depth Rerun inspection contract.

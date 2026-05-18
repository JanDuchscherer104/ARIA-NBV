---
id: 2026-05-18_rerun_selected_view_representation_cleanup
date: 2026-05-18
title: "Rerun Selected View Representation Cleanup"
status: done
topics: [rerun, rollouts, simplification, selected-depth]
confidence: high
canonical_updates_needed: []
files_touched:
  - .configs/rerun_offline.toml
  - aria_nbv/aria_nbv/rerun_inspector/_config.py
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/aria_nbv/rerun_inspector/_rollout_zarr.py
  - aria_nbv/tests/rerun_inspector/test_loggers.py
  - aria_nbv/tests/rerun_inspector/test_rollout_zarr_logger.py
artifacts:
  - .artifacts/rerun/rollouts_v1_smoke_h3_n2_idx000_depth_image.rrd
  - .artifacts/rerun/rollouts_v1_smoke_h3_n2_idx000_point_cloud.rrd
---

## Task

Implemented the Rerun logging cleanup plan for selected rollout views. The
default selected-view representation remains an in-frustum metric depth image,
and `point_cloud` / `both` are now opt-in viewer representations derived from
persisted selected depth, mask, and intrinsics.

## Changes

- Added `rollout_depths.representation`, `max_points`, and `point_radius`.
- Logged selected-depth point clouds as camera-local `Points3D` under the same
  selected candidate camera entity at `.../camera/points`.
- Kept default selected depth images at `.../camera/depth`.
- Moved VIN candidate depth diagnostics from sibling `.../depth` to
  `.../camera/depth`.
- Removed stale Rerun config fields `rollout_plots.log_all_candidates` and
  `primitives.log_mesh`; `primitives.log_gt_mesh` remains the mesh toggle.
- Pruned unused Worker-C/frustum fallback helpers in the Rerun logger.
- Repaired the rollout logger test fixture so target-eval compact candidate
  crop rows stay aligned after tests mutate candidate validity.

## Verification

- `cd aria_nbv && uv run pytest tests/rerun_inspector/test_loggers.py tests/rerun_inspector/test_rollout_zarr_logger.py -q`
  passed with 29 tests.
- `cd aria_nbv && uv run pytest tests/rerun_inspector/test_rerun_cli.py -q`
  passed with 14 tests.
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector tests/rerun_inspector`
  passed.
- Saved smoke RRDs for default depth-image and point-cloud selected-view
  representations, then confirmed serialized entity paths contain only
  `.../camera/depth` for the depth-image artifact and only `.../camera/points`
  for the point-cloud artifact.

## Canonical State Impact

No canonical thesis or backlog update is required. The change is viewer-only:
rollout Zarr schema, selected-depth persistence, masks, candidate ordering, RRI
labels, and training payloads were not changed.

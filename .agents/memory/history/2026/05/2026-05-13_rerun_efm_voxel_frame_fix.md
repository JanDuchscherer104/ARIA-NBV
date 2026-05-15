---
id: 2026-05-13_rerun_efm_voxel_frame_fix
date: 2026-05-13
title: "Rerun EFM Voxel Frame Fix"
status: done
topics: [rerun, frames, efm, visualization]
confidence: high
canonical_updates_needed: []
---

## Task
Fix the Rerun rollout inspector view where EFM voxel evidence appeared outside the GT mesh for the rollout smoke sample.

## Method
Compared the Rerun voxel logger against VIN's existing voxel diagnostics. The Rerun logger treated field indices from tensors shaped `(D,H,W)` as `(x,y,z)`, while the EVL/VIN convention maps `D -> z`, `H -> y`, and `W -> x`. Updated Rerun logging to use that convention and changed the voxel extent from a transformed local box to an explicit world-space oriented `Boxes3D`.

Follow-up inspection showed the remaining large blue slab was not another frame transform bug: `occ_pr` was saturated over most of the 48^3 grid (`>=0.95` for roughly 82% of voxels), so the capped top-k display selected a dense, visually misleading chunk of the local 4 m EFM cube. The inspector now keeps `occ_pr` opt-in and shows the more selective `cent_pr` and `cent_pr_nms` layers by default.

## Verification
Passed:
- `cd aria_nbv && uv run ruff format aria_nbv/rerun_inspector/_loggers.py tests/rerun_inspector/test_loggers.py`
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector/_loggers.py tests/rerun_inspector/test_loggers.py`
- `cd aria_nbv && uv run pytest tests/rerun_inspector/test_loggers.py tests/rerun_inspector/test_rollout_zarr_logger.py -q`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline.toml --rollout-store ../.data/offline_cache/rollouts_v1_smoke.zarr --rollout-index 0 --rollout-context required --save ../.artifacts/rerun/rollout_v1_smoke_frame_fixed.rrd`
- `cd aria_nbv && uv run rerun rrd print ../.artifacts/rerun/rollout_v1_smoke_frame_fixed.rrd | rg 'world/efm/voxels|Boxes3D|Points3D|Transform3D'`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline.toml --rollout-store ../.data/offline_cache/rollouts_v1_smoke.zarr --rollout-index 0 --rollout-context required --save ../.artifacts/rerun/rollout_v1_smoke_frame_fixed_no_occ_pr.rrd`
- `cd aria_nbv && uv run rerun rrd print ../.artifacts/rerun/rollout_v1_smoke_frame_fixed_no_occ_pr.rrd | rg 'world/efm/voxels'`

## Canonical State Impact
None. This is a display-only inspector correction; rollout stores, VIN plotting, and EFM/backbone data remain unchanged.

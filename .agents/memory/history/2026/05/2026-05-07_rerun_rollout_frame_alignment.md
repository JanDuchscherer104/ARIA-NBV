---
id: 2026-05-07_rerun_rollout_frame_alignment
date: 2026-05-07
title: "Rerun Rollout Frame Alignment"
status: done
topics: [rerun, rollout-zarr, geometry, diagnostics]
confidence: high
canonical_updates_needed: []
---

## Task

Make GT mesh transparency configurable in the Rerun offline inspector and
diagnose why synthetic rollout overlays appeared detached from ASE, GT, and EFM
world-space modalities.

## Findings

The visible mismatch came from overlaying a synthetic rollout store whose
candidate centers were near its local smoke-test origin onto a real VIN sample
whose reference pose lived elsewhere in scene world. The store did not persist a
rollout root pose, so the inspector had no authoritative transform from
synthetic rollout coordinates into the selected VIN context.

## Changes

- Added `geometry.mesh_alpha` for GT mesh transparency.
- Added mandatory `rollouts/root_pose_world` to new rollout Zarr stores.
- Rerun rollout inspection now maps a synthetic store root to the selected VIN
  reference pose before logging rollout cameras and centers.
- Regenerated the local synthetic rollout smoke store and `.rrd` artifact with
  the updated schema.

## Verification

- `cd aria_nbv && uv run pytest tests/rerun_inspector tests/data_handling/test_rollout_zarr_store.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector aria_nbv/data_handling/_rollout_zarr_store.py tests/rerun_inspector tests/data_handling/test_rollout_zarr_store.py`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/impl/rerun_offline_inspector.qmd && quarto render contents/impl/one_scene_smoke.qmd && quarto render contents/impl/rollout_storage_contract.qmd`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline.toml --rollout-store ../.artifacts/rerun_rollout_inspector/2026-05-06_multistep/rollouts.zarr --rollout-index 0 --rollout-context required --split val --index 0 --save ../.artifacts/rerun/rollout_with_modalities.rrd`

The regenerated recording contains `world/ase`, `world/gt`, `world/efm`, and
`world/rollout` entities in one recording.

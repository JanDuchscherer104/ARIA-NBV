---
id: 2026-05-04_rerun_inspector_keyframe_display_cleanup
date: 2026-05-04
title: "Rerun Inspector Keyframe Display Cleanup"
status: done
topics: [rerun, offline-store, diagnostics, keyframes]
confidence: high
canonical_updates_needed: []
files_touched:
  - .configs/rerun_offline.toml
  - .configs/rerun_offline_smoke_v6.toml
  - aria_nbv/aria_nbv/rerun_inspector/_config.py
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/tests/rerun_inspector/test_loggers.py
artifacts:
  - .artifacts/rerun/offline_smoke_v6.rrd
---

## Task

Cleaned up the Rerun V2 recording after inspection: rotate ASE keyframe media for display, remove candidate RRI histogram/table outputs, and avoid 2D media under the `world/ase` tree.

## Method

Kept posed ASE camera/frustum entities under `world/ase/cameras/rgb/frame_XXX/camera`, but moved image/depth payloads to `media/ase/cameras/rgb/frame_XXX/{image,depth}`. Applied a display-only 90 degree clockwise rotation to those image/depth arrays before logging.

## Outputs

Regenerated `.artifacts/rerun/offline_smoke_v6.rrd`. The recording contains `media/ase/...` keyframe media and no `/metadata/candidates/table`, `/plots/candidates/rri_histogram`, or `world/ase/.../camera/{image,depth}` media branches.

## Verification

- `cd aria_nbv && uv run pytest tests/rerun_inspector tests/data_handling/test_offline_visual_inventory.py -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector tests/rerun_inspector aria_nbv/data_handling/_offline_visual_inventory.py`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline_smoke_v6.toml --split val --index 0 --save ../.artifacts/rerun/offline_smoke_v6.rrd`
- `cd aria_nbv && uv run rerun rrd print ../.artifacts/rerun/offline_smoke_v6.rrd | rg '/world/ase/cameras|/media/ase|/metadata/candidates|/plots/candidates|/world/ase/cameras/rgb/frame_[0-9]+/camera/(image|depth)'`

## Canonical State Impact

No canonical state files need updates. This is an implementation-level Rerun display cleanup.

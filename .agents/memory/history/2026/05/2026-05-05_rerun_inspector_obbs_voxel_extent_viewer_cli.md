---
id: 2026-05-05_rerun_inspector_obbs_voxel_extent_viewer_cli
date: 2026-05-05
title: "Rerun Inspector OBB Labels, Voxel Extent, And Viewer CLI"
status: done
topics: [rerun, offline-store, geometry, cli]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/aria_nbv/rerun_inspector/_cli.py
  - aria_nbv/tests/rerun_inspector/test_loggers.py
  - aria_nbv/tests/rerun_inspector/test_rerun_cli.py
artifacts:
  - .artifacts/rerun/offline_smoke_v6.rrd
---

## Task

Implemented focused Rerun offline-inspector improvements for EFM voxel extent
visualization, OBB class labels, and one-command viewer launching.

## Method

Logged `world/efm/voxels/extent` as a native `Boxes3D` with an explicit
`ParentFromChild` transform from `backbone_out.t_world_voxel`. Enriched GT and
detected OBB labels with `CompactObbBlock.sem_id_to_name`, preserving
`class=<unknown>` fallback for missing or out-of-range semantic ids.

Added `nbv-rerun-inspect --view` and `--serve-web` post-save foreground viewer
modes. Web serving defaults to localhost and Rerun-selected ports; `--lan`
opts into `0.0.0.0` bind and prints a LAN hint.

## Verification

- `cd aria_nbv && uv run pytest tests/rerun_inspector tests/data_handling/test_offline_visual_inventory.py -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector tests/rerun_inspector aria_nbv/data_handling/_offline_visual_inventory.py`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline_smoke_v6.toml --split val --index 0 --save ../.artifacts/rerun/offline_smoke_v6.rrd`
- `cd aria_nbv && uv run rerun rrd print -vvv ../.artifacts/rerun/offline_smoke_v6.rrd | rg 'class=|EFM voxel extent' -n`

## Canonical State Impact

No canonical project-state update required. This is a scoped inspector and CLI
improvement within the existing Rerun offline-store diagnostic direction.

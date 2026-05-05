---
id: 2026-05-04_rerun_inspector_entity_tree_v2
date: 2026-05-04
title: "Rerun Inspector Entity Tree V2"
status: done
topics: [rerun, offline-store, diagnostics, geometry]
confidence: high
canonical_updates_needed: []
files_touched:
  - .configs/rerun_offline.toml
  - .configs/rerun_offline_smoke_v6.toml
  - aria_nbv/aria_nbv/data_handling/_offline_visual_inventory.py
  - aria_nbv/aria_nbv/rerun_inspector/_cli.py
  - aria_nbv/aria_nbv/rerun_inspector/_config.py
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/tests/rerun_inspector/test_loggers.py
  - aria_nbv/tests/rerun_inspector/test_rerun_cli.py
artifacts:
  - .artifacts/rerun/offline_smoke_v6.rrd
---

## Task

Implemented the Rerun Offline Inspector Entity Tree V2 plan. The inspector now groups ASE observations, GT supervision, candidate diagnostics, EFM evidence, plots, and metadata into separate semantic roots.

## Method

Replaced aggregate candidate line-strip frusta with native per-candidate `Transform3D + Pinhole` camera entities grouped by validity. Added selected-candidate detail logging, candidate table metadata, native RRI histogram, native `Boxes3D` OBBs, thresholded EFM voxel fields, and first/last ASE RGB plus depth keyframes.

## Outputs

Regenerated `.artifacts/rerun/offline_smoke_v6.rrd` from the real `vin_offline_rerun_smoke_v6` offline store. Verified new paths include `world/ase`, `world/gt`, `world/candidates/{valid,invalid}`, `world/efm`, `metadata/candidates`, and `plots/candidates`; legacy `world/candidates/frusta`, `world/mesh`, `world/detected`, `world/semidense`, `world/reference`, and `world/trajectory` paths are absent.

## Verification

- `cd aria_nbv && uv run pytest tests/rerun_inspector tests/data_handling/test_offline_visual_inventory.py -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector tests/rerun_inspector aria_nbv/data_handling/_offline_visual_inventory.py`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline_smoke_v6.toml --split val --index 0 --save ../.artifacts/rerun/offline_smoke_v6.rrd`
- `cd aria_nbv && uv run rerun rrd print ../.artifacts/rerun/offline_smoke_v6.rrd | rg '/world/ase/cameras|/world/gt|/world/efm|/plots|/metadata/candidates'`

## Canonical State Impact

No canonical state files need updates. This is an implementation-level inspector change aligned with the existing Rerun/NBV diagnostics direction.

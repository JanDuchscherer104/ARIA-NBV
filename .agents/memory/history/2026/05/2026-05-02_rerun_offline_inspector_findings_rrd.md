---
id: 2026-05-02_rerun_offline_inspector_findings_rrd
date: 2026-05-02
title: "Rerun Offline Inspector Findings And Smoke RRD"
status: done
topics: [rerun, offline-cache, diagnostics, geometry]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/aria_nbv/rerun_inspector/_metadata.py
  - aria_nbv/tests/rerun_inspector/test_loggers.py
  - .configs/build_vin_offline_rerun_smoke_v6.toml
  - .configs/rerun_offline_smoke_v6.toml
artifacts:
  - .data/offline_cache/vin_offline_rerun_smoke_v6
  - .artifacts/rerun/offline_smoke_v6.rrd
---

Resolved the offline Rerun inspector review findings by preserving explicit
zero-candidate counts, suppressing top-oracle visualization when every masked
candidate is invalid, declaring `world` as right-handed Z-up, and logging
candidate depth diagnostics under per-candidate camera branches with
`Transform3D`, `Pinhole(camera_xyz=LUF)`, and metric `DepthImage` payloads.

The keyframe RGB/depth path now derives camera context from the live EFM
snippet by matching camera frames to trajectory indices and composing
`T_world_cam = T_world_rig @ inverse(T_camera_rig)`. If that context is not
available, the layer is skipped and a metadata warning is recorded instead of
writing orphan image entities.

Added smoke-only configs for a version-6 sidecar offline store and Rerun
recording. The existing `.data/offline_cache/vin_offline` store was left
untouched; the sidecar build wrote one real ASE/EFM sample to
`.data/offline_cache/vin_offline_rerun_smoke_v6`, and the inspector saved
`.artifacts/rerun/offline_smoke_v6.rrd`.

Verification:

- `cd aria_nbv && uv run pytest tests/rerun_inspector tests/data_handling/test_offline_visual_inventory.py -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector tests/rerun_inspector aria_nbv/data_handling/_offline_visual_inventory.py`
- `cd aria_nbv && uv run rerun rrd print ../.artifacts/rerun/offline_smoke_v6.rrd`

---
id: 2026-05-04_rerun_obb_snippet_frame_alignment
date: 2026-05-04
title: "Rerun OBB Snippet Frame Alignment"
status: done
topics: [rerun, offline-store, obb, frames]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/tests/rerun_inspector/test_loggers.py
artifacts:
  - .artifacts/rerun/offline_smoke_v6.rrd
---

## Task
Investigate why GT/detected OBBs in the offline Rerun smoke recording were visibly misaligned with mesh, semidense points, trajectory, and candidate geometry.

## Findings
EFM declares `obbs/padded_snippet` as snippet-coordinate OBBs, and EVL writes `obbs/pred` / `obbs/pred_viz` in snippet coordinates. The Rerun inspector was logging the 34-field `ObbTW` payload directly under `world/*`, so the boxes were displayed in the wrong frame. This was unrelated to `rotate_yaw_cw90`; the missing transform was `T_world_snippet`.

## Changes
The inspector now resolves `snippet/t_world_snippet`, falls back to the first `vin_snippet.t_world_rig` pose when a live EFM snippet is unavailable, transforms OBBs into ARIA world coordinates before line-strip logging, and records a metadata warning instead of drawing unanchored OBBs if no transform can be resolved. A regression test checks that a non-identity snippet transform shifts GT OBB line strips before logging.

## Verification
- `cd aria_nbv && uv run pytest tests/rerun_inspector/test_loggers.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector tests/rerun_inspector aria_nbv/data_handling/_offline_visual_inventory.py`
- `cd aria_nbv && uv run pytest tests/rerun_inspector tests/data_handling/test_offline_visual_inventory.py -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py -q`
- Regenerated `.artifacts/rerun/offline_smoke_v6.rrd`.
- Verified GT OBB line bounds after transform are `[-12.375, -4.802, 0.002]` to `[1.376, 12.711, 2.663]`, matching the mesh bounds `[-12.375, -4.967, -0.05]` to `[1.376, 12.79, 2.703]`.

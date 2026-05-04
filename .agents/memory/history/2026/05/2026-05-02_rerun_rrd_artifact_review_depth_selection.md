---
id: 2026-05-02_rerun_rrd_artifact_review_depth_selection
date: 2026-05-02
title: "Rerun RRD Artifact Review Depth Selection Fix"
status: done
topics: [rerun, diagnostics, offline-cache, candidates]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/tests/rerun_inspector/test_loggers.py
artifacts:
  - .artifacts/rerun/offline_smoke_v6.rrd
---

Reviewed the persisted `.artifacts/rerun/offline_smoke_v6.rrd` recording with
`rerun rrd print` and the Rerun dataframe API. The entity tree was mostly
healthy: `world` declared right-handed Z-up coordinates, candidate frusta and
centers were batched, OBBs/mesh/trajectory were under world-space paths, and
candidate depth entities had parent camera transforms plus `Pinhole` and
metric `DepthImage` components.

The artifact exposed one semantic bug: the top-oracle frustum was candidate
35, but candidate depth branches were logged only for candidates 0 and 59.
Candidate 0 had the highest raw RRI but was invalid, so the depth logger was
using raw `argmax(rri)` instead of the validity-aware top-oracle selector.

Fixed `_log_candidate_depths` to reuse `_top_oracle_index(...)` with the
candidate validity mask. Added a regression assertion that an invalid raw-RRI
maximum is not logged as a selected depth camera. Regenerated
`.artifacts/rerun/offline_smoke_v6.rrd`; it now includes depth/camera branches
for candidates 0, 35, and 59, with candidate 35 matching the top-oracle frustum
label.

Verification:

- `cd aria_nbv && uv run pytest tests/rerun_inspector -q`
- `cd aria_nbv && uv run pytest tests/rerun_inspector tests/data_handling/test_offline_visual_inventory.py -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector tests/rerun_inspector aria_nbv/data_handling/_offline_visual_inventory.py`
- Artifact invariant check via `rerun.dataframe.load_recording(...)` confirmed
  no orphan depth/image branches and expected camera/depth components.

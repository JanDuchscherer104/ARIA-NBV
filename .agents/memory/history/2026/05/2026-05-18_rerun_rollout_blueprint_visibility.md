---
id: 2026-05-18_rerun_rollout_blueprint_visibility
date: 2026-05-18
title: "Rerun Rollout Blueprint Visibility"
status: done
topics: [rerun, rollouts, visualization, blueprint]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/aria_nbv/rerun_inspector/_rollout_zarr.py
  - aria_nbv/tests/rerun_inspector/test_loggers.py
  - aria_nbv/tests/rerun_inspector/test_rollout_zarr_logger.py
artifacts:
  - .artifacts/rerun/rollouts_v1_smoke_scoped_idx000.rrd
---

Task: update the Rerun inspector blueprint so heavy context layers are hidden
by default and rollout step views initially show only selected candidates.

Method: the shared world blueprint keeps `/world/**` as the included content
query and uses Rerun `EntityBehavior(visible=False)` overrides for
`/world/efm/voxels`, `/world/gt/obbs`, and requested rollout subtrees. This
keeps entities present in the Blueprint panel while making them hidden by
default. The rollout Zarr logger sends a second rollout-specific blueprint after
resolving the selected rollout chain; it hides exact `step_*/valid` and
`step_*/invalid` subtrees for the logged rollout/chain/step ids while leaving
`selected` visible.

Verification:
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector tests/rerun_inspector`
- `cd aria_nbv && uv run pytest tests/rerun_inspector -q`
- regenerated `.artifacts/rerun/rollouts_v1_smoke_scoped_idx000.rrd`

Canonical state impact: no durable thesis or backlog update needed; this is a
viewer-default refinement only.

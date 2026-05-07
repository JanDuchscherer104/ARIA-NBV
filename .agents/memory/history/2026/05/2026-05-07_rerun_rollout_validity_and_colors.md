---
id: 2026-05-07_rerun_rollout_validity_and_colors
date: 2026-05-07
title: "Rerun Rollout Validity And Color Diagnostics"
status: done
topics: [rerun, rollouts, validity, visualization]
confidence: high
canonical_updates_needed: []
---

## Task

Jan found that the synthetic multistep rollout overlay showed candidates as
valid even though, after visual alignment onto a real VIN scene, they would
collide with or pass through the mesh.

## Findings

The inspected artifact is a synthetic smoke `rollouts.zarr`. Its stored
candidate validity belongs to the synthetic box scene, not to the explicit VIN
sample used for contextual overlay. Treating those masks as real scene validity
was misleading.

## Changes

- Synthetic rollouts aligned onto an explicit VIN context now display candidate
  validity as untrusted/invalid while retaining the stored validity as metadata.
- Rollout selected paths now start at the root/reference pose and use per-step
  segment colors.
- Rollout candidate centers use per-step colors; invalid synthetic overlays use
  muted step colors.
- GT mesh alpha defaults to roughly 20%.
- GT and detected/predicted OBBs now use separate semantic color families, with
  target-OBB highlighting when a rollout target hint matches available OBB
  labels.

## Verification

- `cd aria_nbv && uv run pytest tests/rerun_inspector -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_rollout_zarr_store.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector tests/rerun_inspector`
- `cd aria_nbv && uv run ruff check aria_nbv/data_handling/_rollout_zarr_store.py tests/data_handling/test_rollout_zarr_store.py`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/impl/rerun_offline_inspector.qmd && quarto render contents/impl/one_scene_smoke.qmd`
- `make check-agent-memory`
- Saved a refreshed `.artifacts/rerun/rollout_with_modalities.rrd` from the
  synthetic rollout store overlaid on `split=val index=0` and inspected the RRD
  printout for `world/gt`, `world/efm`, and untrusted rollout validity fields.

## Canonical State Impact

No canonical state update needed. This is an inspector diagnostic behavior fix,
not a thesis-direction change.

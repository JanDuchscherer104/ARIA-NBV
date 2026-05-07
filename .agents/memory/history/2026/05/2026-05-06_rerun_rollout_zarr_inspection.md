---
id: 2026-05-06_rerun_rollout_zarr_inspection
date: 2026-05-06
title: "Rerun Rollout Zarr Inspection"
status: done
topics: [rerun, rollout, zarr, diagnostics]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rerun_inspector/_cli.py
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/aria_nbv/rerun_inspector/_rollout_zarr.py
  - aria_nbv/tests/rerun_inspector/test_loggers.py
  - aria_nbv/tests/rerun_inspector/test_rerun_cli.py
  - aria_nbv/tests/rerun_inspector/test_rollout_zarr_logger.py
  - docs/contents/impl/rerun_offline_inspector.qmd
  - docs/contents/impl/one_scene_smoke.qmd
artifacts:
  - .artifacts/rerun_rollout_inspector/2026-05-06_multistep/rollouts.zarr
  - .artifacts/rerun_rollout_inspector/2026-05-06_multistep/structural_summary.json
  - .artifacts/rerun_rollout_inspector/2026-05-06_multistep/rollout_multistep_xy_summary.png
  - .artifacts/rerun_rollout_inspector/2026-05-06_multistep/rollout_probability_heatmap.png
  - .artifacts/rerun/rollout_multistep_softmax_2026-05-06.rrd
---

## Task

Integrated standalone multistep `rollouts.zarr` replay inspection into the
existing Rerun inspector and generated synthetic offline rollout samples for
structural and visual stress checks.

## Method

Added `--rollout-store`, `--rollout-index`, and `--rollout-row-id` to
`nbv-rerun-inspect`. The rollout path bypasses VIN sample selection, validates
the replay store, logs one rollout chain on the `rollout_step` timeline, and
uses batched Rerun layers for valid, invalid, and selected candidate frusta,
candidate centers, selected path, scalar plots, and JSON metadata.

Generated a synthetic three-step rollout store with deliberately invalid
unselected candidates. Saved an `.rrd`, printed its Rerun entity inventory, and
created quick PNG summaries for non-viewer visual inspection because the
container has no `DISPLAY`/Wayland session for native Rerun screenshots.

## Outputs

- Store summary: 3 rollouts, 9 steps, 144 candidates, 95 valid rows, 49 invalid
  rows, 9 selected rows.
- Structural checks: validation ok, invalid probability max 0.0, unavailable
  dense Q targets all `NaN`.
- Rerun `.rrd` contains `/world/rollout/candidates/{valid,invalid,selected}`,
  `/world/rollout/selected_path`, `/plots/rollout/*`, and rollout metadata
  entities.

## Verification

- `cd aria_nbv && uv run ruff check aria_nbv/rerun_inspector tests/rerun_inspector`
- `make qmd-frontmatter-check`
- `cd aria_nbv && uv run pytest tests/rerun_inspector tests/data_handling/test_rollout_zarr_store.py -q`
- `cd docs && quarto render contents/impl/rerun_offline_inspector.qmd`
- `cd docs && quarto render contents/impl/one_scene_smoke.qmd`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline.toml --rollout-store ../.artifacts/rerun_rollout_inspector/2026-05-06_multistep/rollouts.zarr --rollout-index 2 --save ../.artifacts/rerun/rollout_multistep_softmax_2026-05-06.rrd`
- `cd aria_nbv && uv run rerun rrd print ../.artifacts/rerun/rollout_multistep_softmax_2026-05-06.rrd`

## Canonical State Impact

No canonical state update is needed. This implements the existing rollout-Zarr
and diagnostic-inspection direction; it does not change thesis scope or stored
decisions.

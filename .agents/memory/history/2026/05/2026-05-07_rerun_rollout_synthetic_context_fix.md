---
id: 2026-05-07_rerun_rollout_synthetic_context_fix
date: 2026-05-07
title: "Rerun Rollout Synthetic Context Fix"
status: done
topics: [rerun, rollout-zarr, diagnostics]
confidence: high
canonical_updates_needed: []
---

## Task

Fix rollout-Zarr Rerun inspection so synthetic multi-step rollout stores can be
overlaid on an explicitly selected VIN offline sample and show the normal
single-step ASE, GT, and EFM modalities.

## Findings

The single-step inspector recording contained `world/ase`, `world/gt`, and
`world/efm` entities. The rollout recording did not because synthetic rollout
stores were kept rollout-only in `auto` mode, and `required` mode could still
prefer synthetic rollout lineage over the explicit VIN sample selector.

## Changes

- Synthetic rollout lineage is now treated as non-authoritative for context
  resolution.
- `--rollout-context required --split val --index 0` now uses the selected VIN
  offline sample as context for synthetic rollout overlays.
- Rerun docs now distinguish rollout-only `auto` smoke from forced VIN-context
  overlays.

## Verification

- `cd aria_nbv && uv run pytest tests/rerun_inspector -q`
- `cd aria_nbv && .venv/bin/ruff check aria_nbv/rerun_inspector/_rollout_zarr.py tests/rerun_inspector/test_rollout_zarr_logger.py`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/impl/rerun_offline_inspector.qmd`
- `cd docs && quarto render contents/impl/one_scene_smoke.qmd`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline.toml --rollout-store ../.artifacts/rerun_rollout_inspector/2026-05-06_multistep/rollouts.zarr --rollout-index 0 --rollout-context required --split val --index 0 --save ../.artifacts/rerun/debug_rollout_required_fixed.rrd`
- `cd aria_nbv && uv run rerun rrd print ../.artifacts/rerun/debug_rollout_required_fixed.rrd | rg '/world/(ase|gt|efm|rollout)'`

The fixed recording contains `world/ase/reference/rig`, `world/ase/semidense`,
`world/gt/mesh`, `world/gt/obbs`, `world/efm/obbs/detected`,
`world/efm/voxels/*`, and `world/rollout/*`.

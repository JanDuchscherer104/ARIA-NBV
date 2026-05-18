---
id: 2026-05-18_selected_depth_rollout_persistence
date: 2026-05-18
title: "Selected Depth Rollout Persistence"
status: done
topics: [rollouts, zarr, q_h, rendering, data-generation]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rendering/candidate_depth_renderer.py
  - aria_nbv/aria_nbv/pose_generation/counterfactuals.py
  - aria_nbv/aria_nbv/rollouts/dataset_writer.py
  - aria_nbv/aria_nbv/rollouts/zarr_store.py
  - aria_nbv/aria_nbv/data_handling/README.md
  - .agents/references/rollout_zarr_q_invalidity_contract.md
  - docs/typst/thesis/advisor_distillation.typ
---

## Task

Implemented selected-action depth persistence for multi-step rollout stores.
The chosen contract is `240 x 240`, metric metres, `float16` depth with invalid
pixels filled by `0.0`, a separate boolean valid mask, and Zarr v3
Blosc/Zstd level 5 bitshuffle chunks by selected-step rows.

## Method

The all-candidate oracle RRI path remains on `target_scorer.depth` and its
low-resolution render settings. A new selected-depth renderer config derives
from that scorer renderer but renders only selected compact-valid candidate
indices at exact output size. `CounterfactualStepResult` now carries optional
selected-depth raster and camera metadata, and `RolloutDatasetWriter` attaches
one selected-depth render per materialized retained step before writing records.

`rollouts.zarr` schema was bumped to `0.5-selected-depth`. The new
`selected_depth/` group stores step/candidate row backlinks, compressed
`depth_m`, compressed `valid_mask`, and selected camera metadata. Validation
requires one selected-depth row per rollout step when selected depth is enabled.
`q_h_view()` continues to read only light metadata and does not eagerly load
the depth rasters.

## Verification

- `cd aria_nbv && uv run ruff format ...`
- `cd aria_nbv && uv run ruff check ...`
- `cd aria_nbv && uv run pytest tests/rollouts`
- `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py`
- `cd aria_nbv && uv run pytest tests/rendering`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`
- `cd docs && typst compile typst/thesis/advisor_distillation.typ --root . /tmp/advisor_distillation_selected_depth.pdf`
- `git diff --check`

The KG claim check for the advisor-facing storage claim ran on 2026-05-18 and
returned `unverifiable` because the current literature KG lacks source paths
for this code-level claim.

## Canonical State Impact

No canonical state files need updates. The durable contract is captured in the
rollout Zarr reference, data-handling README, smoke/LRZ configs, and advisor
distillation text.

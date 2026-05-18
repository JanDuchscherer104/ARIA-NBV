---
id: 2026-05-18_rollout_schema_06_core
date: 2026-05-18
title: "Rollout Schema 0.6 Rollout Core"
status: done
topics: [rollouts, zarr, q-h, data-generation, simplification]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rollouts/zarr_store.py
  - aria_nbv/aria_nbv/rerun_inspector/_rollout_zarr.py
  - aria_nbv/tests/rollouts/test_zarr_store.py
  - aria_nbv/tests/rerun_inspector/test_rollout_zarr_logger.py
  - aria_nbv/aria_nbv/data_handling/README.md
  - docs/reference/aria_nbv.rollouts.RolloutZarrStoreReader.qmd
  - docs/reference/aria_nbv.rollouts.zarr_store.RolloutZarrStoreReader.qmd
  - docs/typst/shared/data-layout-trees.typ
  - docs/figures/diagrams/data_handling/mermaid/offline_rollout_physical_layout.mmd
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/references/rollout_zarr_q_invalidity_contract.md
---

## Task

Implemented rollout Zarr schema `0.6-rollout-core` with no stale-store
migration path.

## Outputs

- Removed redundant candidate and step arrays:
  `candidate_valid_mask`, `padded_mask`, `heavy_diag_available_mask`,
  `selection_entropy`, `steps/transition_id`, and `dictionaries/transition`.
- Made `candidates/actor_action_mask` the canonical persisted action-feasibility
  mask.
- Reduced `q_h/` to one-step target labels plus selected-transition TD arrays
  and added `td_semantics="selected_transition_only"`.
- Updated Rerun rollout logging to derive entropy from persisted probabilities
  and log only the canonical validity mask.
- Regenerated `.data/offline_cache/rollouts_v1_smoke.zarr` under schema
  `0.6-rollout-core`.

## Verification

- `cd aria_nbv && uv run pytest tests/rollouts/test_zarr_store.py tests/rollouts/test_dataset_writer.py -q`
  passed: 31 passed.
- `cd aria_nbv && uv run pytest tests/rerun_inspector/test_rollout_zarr_logger.py tests/app/panels/test_counterfactual_rollouts_panel.py -q`
  passed: 29 passed.
- `cd aria_nbv && uv run ruff check aria_nbv/rollouts/zarr_store.py aria_nbv/rerun_inspector/_rollout_zarr.py tests/rollouts/test_zarr_store.py tests/rerun_inspector/test_rollout_zarr_logger.py`
  passed.
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`
  passed.
- `cd aria_nbv && uv run nbv-rollouts-info --store ../.data/offline_cache/rollouts_v1_smoke.zarr --validate --json`
  validated the regenerated schema `0.6-rollout-core` store.

## Canonical State Impact

Agents DB and the rollout Q/invalidity contract now describe schema
`0.6-rollout-core`, selected-transition Q_H semantics, and the removed arrays.

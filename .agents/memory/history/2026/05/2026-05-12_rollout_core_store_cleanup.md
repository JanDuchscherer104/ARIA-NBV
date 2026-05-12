---
id: 2026-05-12_rollout_core_store_cleanup
date: 2026-05-12
title: "Rollout Core Store Cleanup"
status: done
topics: [aria-nbv, rollouts, zarr, simplification]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - aria_nbv/aria_nbv/rollouts/
  - aria_nbv/aria_nbv/data_handling/_target_selection.py
  - aria_nbv/aria_nbv/pose_generation/counterfactuals.py
  - aria_nbv/aria_nbv/pose_generation/target_counterfactuals.py
  - aria_nbv/aria_nbv/data_handling/README.md
  - aria_nbv/tests/rollouts/
---

## Task

Implement the rollout core simplification around one production source
contract: `VinOfflineSample` roots and standalone factual `rollouts.zarr`
stores. Remove debug-only trace persistence, stop storing derived `q_h/`
arrays, and keep inline TODOs unless their exact issue was resolved.

## Outcome

- Locked `ActorVisibleTargetSelector` and rollout generation to
  `VinOfflineSample`; `VinOracleBatch` is no longer accepted as a rollout root.
- Bumped rollout Zarr to `0.3-source-facts-derived-qh`, added a shared
  `sources/` table, removed persisted `q_h/`, and exposed
  `RolloutZarrStoreReader.q_h_view()` as the derived finite-candidate view.
- Removed the production MessagePack smoke CLI/export path. Synthetic traces
  now live in test fixtures.
- Replaced generic counterfactual metric dictionaries at the scorer boundary
  with `CounterfactualMetricBundle` while keeping compatibility accessors for
  existing trace code.
- Updated Rerun, Streamlit, tests, README guidance, and canonical decisions to
  reflect factual replay storage plus derived Q views.

## Verification

- `cd aria_nbv && uv run ruff format <touched files>`: passed.
- `cd aria_nbv && uv run ruff check <touched files>`: passed.
- `cd aria_nbv && uv run ruff check --select F401,TCH <touched files>`: passed.
- `cd aria_nbv && uv run pytest tests/rollouts tests/data_handling/test_target_selection.py tests/data_handling/test_public_api_contract.py tests/pose_generation/test_counterfactuals.py tests/rerun_inspector tests/app/panels -q`: 116 passed, 2 warnings.
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`: passed.

## Notes

Old rollout Zarr shards are intentionally not backward-compatible with schema
`0.3-source-facts-derived-qh`. The store now records replay facts; padded
`Q_H` tensors belong to reader/training views until scale profiling says
otherwise.

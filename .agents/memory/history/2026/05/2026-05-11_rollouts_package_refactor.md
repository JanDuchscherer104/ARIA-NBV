---
id: 2026-05-11_rollouts_package_refactor
date: 2026-05-11
title: "Rollouts Package Refactor"
status: done
topics: [aria-nbv, rollouts, data-handling, package-boundary]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/AGENTS.md
  - aria_nbv/aria_nbv/rollouts/
  - aria_nbv/aria_nbv/data_handling/__init__.py
  - aria_nbv/aria_nbv/pose_generation/__init__.py
  - aria_nbv/tests/rollouts/
---

## Task

Move multi-step rollout trace, replay-store, writer, and CLI ownership out of
`aria_nbv.data_handling` / `aria_nbv.pose_generation` into the dedicated
`aria_nbv.rollouts` package.

## Outcome

- `aria_nbv.rollouts` is now the canonical public import surface for
  `RolloutTrace`, `RolloutStepTrace`, rollout invalidity codes,
  `RolloutDatasetWriterConfig`, `RolloutZarrStoreReader`, and
  `write_rollout_zarr_store`.
- `aria_nbv.data_handling` no longer exports rollout writer/Zarr contracts and
  remains focused on raw snippets, VIN offline stores, VIN batches, and target
  selection.
- `aria_nbv.pose_generation` no longer exports rollout trace persistence.
- Added `aria_nbv/aria_nbv/rollouts/AGENTS.md` and updated the package guide so
  future rollout replay work routes to the new owner.

## Verification

- `cd aria_nbv && uv run ruff check ...`: passed for the touched Python files.
- `cd aria_nbv && uv run pytest tests/rollouts tests/data_handling/test_public_api_contract.py tests/data_handling/test_target_selection.py tests/rerun_inspector/test_rollout_zarr_logger.py tests/rerun_inspector/test_rerun_cli.py tests/app/panels tests/pose_generation/test_counterfactuals.py tests/test_config_field_constraints.py -q`: 111 passed.
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`: passed.

## Notes

No backward-compatible rollout exports were kept on `aria_nbv.data_handling` or
`aria_nbv.pose_generation`; this was intentional to make the separation visible.
Generated API reference pages were not regenerated because `quartodoc` is not
available in the current package environment; `docs/_quarto.yml` now points
future generation at `rollouts.*`.

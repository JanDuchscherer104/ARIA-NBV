---
id: 2026-05-12_rollout_data_handling_simplification
date: 2026-05-12
title: "Rollout Data Handling Simplification"
status: done
topics: [aria-nbv, rollouts, data-handling, simplification]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rollouts/
  - aria_nbv/aria_nbv/data_handling/
  - aria_nbv/aria_nbv/utils/
  - aria_nbv/tests/rollouts/
---

## Task

Streamline the active rollout data path after the `aria_nbv.rollouts` package
split, while preserving rollout labels, masks, target validity, and the
`0.2-clean-tables` Zarr schema.

## Outcome

- Removed stale VIN runtime alias plumbing: `_vin_runtime.py`,
  `_sample_keys.py`, and `_config_utils.py` are gone, and
  `VinOnlineDatasetConfig` is no longer exported.
- Added shared config-path and fingerprint helpers under `aria_nbv.utils`
  without exporting the path helper through `aria_nbv.utils.__init__`, avoiding
  a `configs`/`utils` import cycle.
- Replaced loose rollout writer hash helpers with `RolloutSourceLineageBuilder`.
- Replaced target-selector sample-view construction with explicit adapters for
  `VinOfflineSample` and `VinOracleBatch`.
- Added rollout Zarr write/validation method objects while keeping public store
  functions as thin facades.

## Verification

- `make loc`: blocked because the repo has no `loc` target.
- `ruff format` and `ruff check` on touched Python surfaces: passed.
- `ruff check --select F401,TCH` on touched Python surfaces: passed.
- `cd aria_nbv && uv run pytest tests/rollouts tests/data_handling/test_public_api_contract.py tests/data_handling/test_target_selection.py`: 34 passed.
- `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py tests/rerun_inspector/test_rollout_zarr_logger.py tests/rerun_inspector/test_rerun_cli.py tests/app/panels`: 54 passed.
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`: passed.

## Notes

Unresolved inline TODOs outside the active rollout path were preserved. A
Pydantic config annotation requires `pathlib.Path` at runtime in
`_offline_store.py`; it is intentionally exempted from TCH movement.

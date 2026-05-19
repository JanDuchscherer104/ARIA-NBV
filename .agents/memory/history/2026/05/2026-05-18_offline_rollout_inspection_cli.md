---
id: 2026-05-18_offline_rollout_inspection_cli
date: 2026-05-18
title: "Offline And Rollout Inspection CLI"
status: done
topics: [vin-offline-store, rollouts, cli, rerun, makefile]
confidence: high
canonical_updates_needed: []
files_touched:
  - Makefile
  - aria_nbv/pyproject.toml
  - aria_nbv/aria_nbv/data_handling/offline_info_cli.py
  - aria_nbv/aria_nbv/rollouts/info_cli.py
  - aria_nbv/tests/data_handling/test_offline_info_cli.py
  - aria_nbv/tests/rollouts/test_info_cli.py
artifacts:
  - .artifacts/rerun/offline_random_cli_smoke.rrd
  - .artifacts/rerun/rollout_random_cli_smoke.rrd
---

## Task

Implemented read-only operator inspection for immutable VIN offline stores and
standalone rollout Zarr stores. Added package CLIs and Makefile wrappers for
summary, tree, sampled rows, deterministic random row selection, and Rerun
inspection handoff.

## Method

Added `nbv-offline-info` with `summary`, `tree`, `samples`, and
`random-index` subcommands. Extended `nbv-rollouts-info` with `--stats` and
`--random-index --min-horizon --seed` while preserving the existing default
manifest output. Added repo-root Make targets for offline and rollout store
inspection, including save/view Rerun modes that delegate to
`nbv-rerun-inspect`.

## Findings

The Make Rerun wrappers must pass absolute store paths to `nbv-rerun-inspect`.
Passing a repo-relative `.data/offline_cache/...` path back through the
inspector config validator double-resolved under `.data/offline_cache`; the
wrapper now normalizes store paths with `realpath -m` before handoff.

## Verification

Ran focused CLI tests:
`cd aria_nbv && uv run pytest tests/data_handling/test_offline_info_cli.py tests/rollouts/test_info_cli.py -q`.

Ran lint:
`cd aria_nbv && uv run ruff check aria_nbv/data_handling aria_nbv/rollouts tests/data_handling tests/rollouts`.

Ran operator smokes from the repo root:
`make offline-info`, `make offline-tree`, `make rollouts-stats`, and
`make offline-rerun-random RERUN_MODE=save OFFLINE_SEED=0 RERUN_SAVE=../.artifacts/rerun/offline_random_cli_smoke.rrd`,
and
`make rollouts-rerun-random RERUN_MODE=save ROLLOUT_SEED=0 ROLLOUT_RERUN_SAVE=../.artifacts/rerun/rollout_random_cli_smoke.rrd`.

## Canonical State Impact

No canonical thesis-state updates are required. This is an operator ergonomics
and inspection-surface addition; it does not change VIN store, rollout store, or
RRI semantics.

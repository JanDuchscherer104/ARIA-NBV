---
id: 2026-05-18_typer_rich_cli_migration
date: 2026-05-18
title: "Typer Rich CLI Migration"
status: done
topics: [cli, data-handling, rollouts, rerun, tooling]
confidence: high
canonical_updates_needed: []
artifacts:
  - .artifacts/rerun/offline_random.rrd
---

## Task

Migrated current argparse package CLIs to Typer while preserving existing console
script names and operator workflows. Added Rich human-facing output for offline
and rollout inspection commands, with JSON output left raw for scripts.

## Method

Added Typer as a package dependency and introduced small shared helpers for
Rich CLI formatting and Typer console-script wrappers. Converted
`nbv-build-offline`, `nbv-offline-info`, `nbv-build-rollouts`,
`nbv-plan-rollout-shards`, `nbv-status-rollout-shards`, `nbv-rollouts-info`,
and `nbv-rerun-inspect`.

Compatibility details preserved:

- `nbv-offline-info` defaults to `summary` through the console wrapper.
- `random-index` text output remains a bare integer.
- `--json` output remains machine-readable JSON without Rich rendering.
- `nbv-rerun-inspect --save` and `--connect` keep optional-value behavior.
- Rollout build manifest lineage still receives the original raw argv.

## Verification

- `cd aria_nbv && uv run pytest tests/data_handling/test_offline_info_cli.py tests/data_handling/test_offline_cli.py tests/rollouts/test_info_cli.py tests/rollouts/test_cli_typer.py tests/rerun_inspector/test_rerun_cli.py -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_offline_info_cli.py tests/rollouts/test_info_cli.py tests/rerun_inspector tests/rollouts -q`
- `cd aria_nbv && uv run ruff check aria_nbv/data_handling aria_nbv/rollouts aria_nbv/rerun_inspector aria_nbv/utils tests/data_handling tests/rollouts tests/rerun_inspector`
- `make offline-info`
- `make offline-tree`
- `make offline-samples`
- `make rollouts-stats`
- `make offline-rerun-random RERUN_MODE=save`

## State Impact

No canonical state update is needed. This is an implementation/tooling change
that keeps existing CLI names and semantics stable while improving human
operator output.

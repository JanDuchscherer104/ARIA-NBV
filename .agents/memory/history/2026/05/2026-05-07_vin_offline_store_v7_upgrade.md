---
id: 2026-05-07_vin_offline_store_v7_upgrade
date: 2026-05-07
title: "VIN Offline Store V7 Upgrade"
status: done
topics: [data, offline-store, simplification, rerun, docs]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - aria_nbv/aria_nbv/data_handling/_offline_store.py
  - aria_nbv/aria_nbv/data_handling/_offline_format.py
  - aria_nbv/aria_nbv/data_handling/_offline_dataset.py
  - aria_nbv/aria_nbv/data_handling/_offline_writer.py
  - aria_nbv/aria_nbv/data_handling/_vin_sources.py
  - aria_nbv/aria_nbv/data_handling/_offline_diagnostics.py
  - aria_nbv/aria_nbv/lightning/lit_datamodule.py
  - aria_nbv/aria_nbv/rerun_inspector/_config.py
  - aria_nbv/aria_nbv/rerun_inspector/_cli.py
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
  - aria_nbv/tests/data_handling/test_public_api_contract.py
  - .configs/build_vin_offline_81286.toml
  - .configs/build_vin_offline_rerun_smoke_v7.toml
  - .configs/rerun_offline.toml
  - .configs/rerun_offline_smoke_v7.toml
  - SETUP.md
  - docs/contents/setup.qmd
  - docs/contents/impl/one_scene_smoke.qmd
  - docs/contents/thesis/m1_contract_report.qmd
  - .agents/refactors.toml
  - .agents/resolved.toml
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
artifacts:
  - .data/offline_cache/vin_offline/manifest.json
  - .data/offline_cache/vin_offline_rerun_smoke_v7/manifest.json
---

## Task

Upgrade the current persisted VIN offline stores to the newest strict dataset
format with no backward compatibility and no retained migration functionality.

## Method

Bumped `OFFLINE_DATASET_VERSION` from 6 to 7 and removed the premature
counterfactual offline-store schema/config surface. The store no longer exposes
`VinOfflineCounterfactuals`, `materialized_blocks.counterfactuals`,
`include_counterfactuals`, `load_counterfactuals`,
`load_counterfactuals_for_batch`, or `_build_counterfactuals`.

Updated the local persisted `.data/offline_cache/vin_offline` and one-sample
Rerun sidecar manifests in place: both now have version 7 and no counterfactual
placeholder fields. The sidecar path and configs were renamed from
`vin_offline_rerun_smoke_v6` to `vin_offline_rerun_smoke_v7`. The one-off
artifact update ran from an inline command and left no migration script or
reader compatibility path in the repo.

## Verification

- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py`
- `cd aria_nbv && uv run nbv-summary --config-path offline_only.toml`
- `cd aria_nbv && .venv/bin/ruff format ...`
- `cd aria_nbv && .venv/bin/ruff check ...`
- `cd aria_nbv && uv run pytest tests/rerun_inspector -q`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/setup.qmd`
- `cd docs && quarto render contents/impl/one_scene_smoke.qmd`
- `cd docs && quarto render contents/thesis/m1_contract_report.qmd`
- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`

## Notes

The default `vin_offline` store still has `stats.interrupted = true` and remains
diagnostic evidence, not final training-scale evidence. The version gate is no
longer the blocker: strict v7 readers open the store.

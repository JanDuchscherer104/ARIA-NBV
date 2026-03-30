---
id: 2026-03-30_data_handling_legacy_api_split
date: 2026-03-30
title: "Data Handling Legacy API Split"
status: done
topics: [data-handling, legacy-cache, migration, api, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - aria_nbv/aria_nbv/data_handling/__init__.py
  - aria_nbv/aria_nbv/data_handling/README.md
  - aria_nbv/aria_nbv/data_handling/_legacy_cache_api.py
  - aria_nbv/aria_nbv/data_handling/_legacy_offline_cache_coverage.py
  - aria_nbv/aria_nbv/data_handling/_legacy_offline_cache_serialization.py
  - aria_nbv/aria_nbv/data_handling/_legacy_offline_cache_store.py
  - aria_nbv/aria_nbv/data_handling/_legacy_oracle_cache.py
  - aria_nbv/aria_nbv/data_handling/_legacy_vin_cache.py
  - aria_nbv/aria_nbv/data_handling/_legacy_vin_provider.py
  - aria_nbv/aria_nbv/data_handling/_legacy_vin_source.py
  - aria_nbv/aria_nbv/data_handling/_migration.py
  - aria_nbv/aria_nbv/data_handling/_vin_runtime.py
  - aria_nbv/aria_nbv/data_handling/_vin_sources.py
  - aria_nbv/aria_nbv/data_handling/offline_cache_coverage.py
  - aria_nbv/aria_nbv/data_handling/offline_cache_serialization.py
  - aria_nbv/aria_nbv/data_handling/offline_cache_store.py
  - aria_nbv/aria_nbv/data_handling/oracle_cache.py
  - aria_nbv/aria_nbv/data_handling/vin_cache.py
  - aria_nbv/aria_nbv/data_handling/vin_oracle_datasets.py
  - aria_nbv/aria_nbv/data_handling/vin_provider.py
  - aria_nbv/aria_nbv/app/app.py
  - aria_nbv/aria_nbv/app/panels/offline_cache_utils.py
  - aria_nbv/aria_nbv/app/panels/offline_stats.py
  - aria_nbv/aria_nbv/app/panels/testing_attribution.py
  - aria_nbv/aria_nbv/app/panels/vin_diagnostics.py
  - aria_nbv/aria_nbv/app/panels/vin_utils.py
  - aria_nbv/aria_nbv/lightning/aria_nbv_experiment.py
  - aria_nbv/aria_nbv/lightning/cli.py
  - aria_nbv/aria_nbv/lightning/lit_datamodule.py
  - aria_nbv/tests/data/test_offline_cache_split.py
  - aria_nbv/tests/data/test_vin_snippet_cache_datamodule_equivalence.py
  - aria_nbv/tests/data/test_vin_snippet_cache_real_data.py
  - aria_nbv/tests/data_handling/test_cache_v2.py
  - aria_nbv/tests/data_handling/test_public_api_contract.py
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
  - aria_nbv/tests/lightning/test_vin_datamodule_sources.py
  - .agents/memory/state/PROJECT_STATE.md
assumptions:
  - The legacy oracle-cache and VIN-snippet-cache runtime still has to remain usable during the offline-store cutover, but it no longer needs to share the canonical package-root API.
---

Task: separate the canonical `aria_nbv.data_handling` API from the remaining legacy oracle-cache / VIN-snippet-cache path while keeping backward-compatible submodule imports working through dedicated compatibility files.

Method: moved the legacy cache implementation behind dedicated `_legacy_*` modules, introduced `_legacy_cache_api.py` and `_legacy_vin_source.py` as grep-visible compatibility owners, reduced `_vin_sources.py` to the canonical online/offline dataset-source surface, and converted the old legacy public submodules (`oracle_cache.py`, `vin_cache.py`, `vin_provider.py`, `offline_cache_*`, `vin_oracle_datasets.py`) into thin compatibility wrappers. Updated app, Lightning, and affected tests to import legacy functionality explicitly from the `_legacy_*` surfaces instead of from the canonical package root.

Findings:
- The package root needed lazy exports for non-raw symbols to avoid an existing import cycle through `pose_generation` during package initialization.
- The monkeypatch-heavy cache tests required the public legacy submodules to alias the real implementation modules, not merely re-export names.
- `vin_oracle_datasets.py` was a useful place to preserve the old broad source-union import path while keeping the canonical owner in `_vin_sources.py`.

Verification:
- `cd aria_nbv && .venv/bin/ruff format <touched files>`
- `cd aria_nbv && .venv/bin/ruff check <touched files>`
- `cd aria_nbv && .venv/bin/python - <<'PY' ... import aria_nbv.data_handling, legacy wrapper submodules, and aria_nbv.streamlit_app ... PY`
- `cd aria_nbv && uv run pytest --capture=no tests/data_handling/test_public_api_contract.py tests/data_handling/test_cache_v2.py tests/data_handling/test_vin_offline_store.py tests/lightning/test_vin_datamodule_sources.py tests/data/test_offline_cache.py tests/data/test_offline_cache_pruning.py tests/data/test_offline_cache_split.py tests/data/test_offline_cache_writer_skip.py tests/data/test_offline_cache_coverage.py tests/data/test_vin_snippet_cache.py tests/data/test_vin_snippet_cache_datamodule_equivalence.py tests/data/test_vin_snippet_cache_real_data.py`
- Result: `41 passed, 7 skipped`
- `cd aria_nbv && timeout 20s uv run nbv-st`
- Streamlit startup reached `Local URL: http://localhost:8501` before the timeout stopped the process.

Canonical state impact: `PROJECT_STATE.md` now records that the `aria_nbv.data_handling` package root is canonical-only and that the remaining legacy cache API lives behind dedicated `_legacy_*` compatibility modules.

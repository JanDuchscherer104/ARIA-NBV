---
id: 2026-03-30_legacy_cache_cutover_markers
date: 2026-03-30
title: "Legacy Cache Cutover Markers"
status: done
topics: [data-handling, migration, cleanup, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - aria_nbv/aria_nbv/data_handling/README.md
  - aria_nbv/aria_nbv/data_handling/__init__.py
  - aria_nbv/aria_nbv/data_handling/_migration.py
  - aria_nbv/aria_nbv/data_handling/offline_cache_coverage.py
  - aria_nbv/aria_nbv/data_handling/offline_cache_serialization.py
  - aria_nbv/aria_nbv/data_handling/offline_cache_store.py
  - aria_nbv/aria_nbv/data_handling/oracle_cache.py
  - aria_nbv/aria_nbv/data_handling/vin_cache.py
  - aria_nbv/aria_nbv/data_handling/vin_oracle_datasets.py
  - aria_nbv/aria_nbv/data_handling/vin_provider.py
  - aria_nbv/aria_nbv/lightning/aria_nbv_experiment.py
  - aria_nbv/aria_nbv/lightning/cli.py
  - aria_nbv/aria_nbv/lightning/lit_datamodule.py
  - aria_nbv/aria_nbv/app/app.py
  - aria_nbv/aria_nbv/app/panels/offline_cache_utils.py
  - aria_nbv/aria_nbv/app/panels/offline_stats.py
  - aria_nbv/aria_nbv/app/panels/testing_attribution.py
  - aria_nbv/aria_nbv/app/panels/vin_diagnostics.py
  - aria_nbv/aria_nbv/app/panels/vin_utils.py
  - aria_nbv/tests/data/test_offline_cache.py
  - aria_nbv/tests/data/test_offline_cache_pruning.py
  - aria_nbv/tests/data/test_offline_cache_split.py
  - aria_nbv/tests/data/test_offline_cache_writer_skip.py
  - aria_nbv/tests/data/test_vin_snippet_cache.py
  - aria_nbv/tests/data/test_vin_snippet_cache_datamodule_equivalence.py
  - aria_nbv/tests/data/test_vin_snippet_cache_real_data.py
  - aria_nbv/tests/data_handling/test_cache_v2.py
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
  - aria_nbv/tests/lightning/test_vin_datamodule_sources.py
  - .agents/memory/state/PROJECT_STATE.md
assumptions:
  - The removable legacy slice is the oracle-cache plus VIN-snippet-cache path and the runtime/UI/CLI/tests that still depend on it.
---

Task: mark all remaining legacy oracle-cache and VIN-snippet-cache code so the final removal sweep is easy once the immutable VIN offline store becomes the only offline path.

Method: added one grep-stable marker string, `NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION`, to the remaining legacy runtime modules, mixed runtime branches that still opt into the old cache path, and the dedicated tests. Documented the marker in `aria_nbv/aria_nbv/data_handling/README.md` and recorded the convention in `.agents/memory/state/PROJECT_STATE.md`.

Findings: the remaining old-path surface was concentrated in `data_handling/{oracle_cache,vin_cache,vin_provider,offline_cache_*}`, the cached-source branch in `vin_oracle_datasets.py`, several Streamlit and Lightning entry points, and the dedicated legacy cache tests. A post-edit audit found no remaining runtime or test files that reference `OracleRriCacheDatasetConfig`, `VinOracleCacheDatasetConfig`, `VinSnippetCacheConfig`, or `OracleRriCacheConfig` without the cutover marker.

Verification:
- `cd aria_nbv && .venv/bin/ruff format ...`
- `cd aria_nbv && .venv/bin/ruff check ...`
- `cd aria_nbv && .venv/bin/python -m pytest --capture=no tests/data/test_offline_cache_split.py tests/data/test_offline_cache_pruning.py tests/data/test_offline_cache_writer_skip.py tests/data/test_vin_snippet_cache.py tests/data_handling/test_cache_v2.py tests/data_handling/test_vin_offline_store.py tests/lightning/test_vin_datamodule_sources.py`
- `make check-agent-memory`
- marker audit: no remaining runtime/test files with the legacy cache config types lacked `NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION`

Canonical state impact: `PROJECT_STATE.md` now records the grep-stable cutover marker used for the final legacy cache removal sweep.

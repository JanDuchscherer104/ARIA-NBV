---
id: 2026-03-30_remove_legacy_data_mirrors
date: 2026-03-30
title: "Removed legacy data mirror modules and canonicalized plotting imports"
status: done
topics: [data-handling, plotting, vin, tests, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - aria_nbv/aria_nbv/data/__init__.py
  - aria_nbv/aria_nbv/data_handling/__init__.py
  - aria_nbv/aria_nbv/utils/data_plotting.py
  - aria_nbv/aria_nbv/vin/__init__.py
  - aria_nbv/tests/data_handling/test_public_api_contract.py
---

## Task
Remove the remaining deprecated `aria_nbv.data` mirror modules, move the shared snippet plotting owner out of the deprecated package, and clean up runtime imports to use canonical `aria_nbv.data_handling` or `aria_nbv.utils` surfaces only.

## Method
Moved the plotting implementation from `aria_nbv.data.plotting` to `aria_nbv.utils.data_plotting`, updated runtime and test imports, copied cache-coverage and serialization helpers into `aria_nbv.data_handling`, deleted the mirrored `aria_nbv.data.*` modules, and tightened the public-surface contract tests to assert that those legacy modules are gone. Kept `vin/__init__.py` explicit, then resolved the resulting import cycle by pointing `vin/model_v3.py` at the canonical `data_handling` owner submodules directly.

## Findings
The strict removal pass was blocked by a real cycle between `aria_nbv.data_handling` initialization and the eager `aria_nbv.vin` package root. The minimal stable fix was to keep explicit `vin/__init__.py` exports while allowing `vin/model_v3.py` to import directly from `data_handling._raw` and `data_handling.vin_adapter`. Streamlit app imports also required restoring hidden root availability for `OracleRriCacheSample` while keeping it off `data_handling.__all__`.

## Verification
Ran `ruff format` and `ruff check` on the touched runtime and test files. Ran `uv run pytest --capture=no tests/data_handling/test_public_api_contract.py tests/data_handling/test_dataset.py tests/data_handling/test_mesh_cache.py tests/data/test_offline_cache_coverage.py tests/data/test_vin_snippet_cache.py tests/test_plotting_frustum_device.py tests/test_plotting_helpers_refactor.py tests/test_plotting_semidense.py` and got `34 passed, 1 skipped`. Ran `.venv/bin/python -c "from aria_nbv.streamlit_app import streamlit_entry; print('streamlit-import-ok')"` successfully.

## Canonical State Impact
Updated `.agents/memory/state/PROJECT_STATE.md` to record that shared snippet plotting now lives in `aria_nbv.utils.data_plotting` alongside the already-canonical `aria_nbv.data_handling` contracts.

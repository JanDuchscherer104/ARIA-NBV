---
id: 2026-03-30_data_surface_canonicalization
date: 2026-03-30
title: "Canonicalized legacy data surface onto data_handling"
status: done
topics: [data-handling, compatibility, caches, plotting, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
---

## Task
Collapse the live `aria_nbv.data` vs `aria_nbv.data_handling` split for the mirrored raw/cache modules, remove the dead legacy Lightning module, and verify the remaining migration claims against the current implementation.

## Method
Replaced mirrored legacy modules under `aria_nbv.data` with compatibility aliases that bind each old import path to the canonical `aria_nbv.data_handling` module object. Extracted shared cache/store path resolution into `PathConfig.resolve_cache_artifact_dir()`, updated canonical cache configs to use it, moved shared `_pretty_label()` into `aria_nbv.utils.plotting`, and switched straightforward runtime imports onto the `aria_nbv.data_handling` root API.

## Findings
Module aliasing was the correct compatibility mechanism because several legacy tests monkeypatch module-level globals on `aria_nbv.data.offline_cache` and `aria_nbv.data.vin_snippet_cache`; thin re-export stubs would not have preserved those semantics. `offline_cache_coverage.py` remains a live old-surface owner and was intentionally left in place. The remaining `aria_nbv.data.*` runtime imports are limited to plotting helpers and old coverage/internal helper surfaces that are not yet root-exported from `data_handling`.

## Verification
Ran `aria_nbv/.venv/bin/ruff format ...` and `aria_nbv/.venv/bin/ruff check ...` on the touched Python files. Ran `aria_nbv/.venv/bin/python -m pytest --capture=no tests/data_handling/test_public_api_contract.py tests/test_pathconfig_isolation_regression.py tests/data_handling/test_vin_offline_store.py tests/data_handling/test_cache_v2.py tests/data_handling/test_dataset.py tests/data_handling/test_mesh_cache.py tests/data/test_efm_snippet_loader.py tests/data/test_offline_cache.py tests/data/test_offline_cache_pruning.py tests/data/test_offline_cache_split.py tests/data/test_offline_cache_writer_skip.py tests/data/test_vin_snippet_cache.py` from `aria_nbv/` and got `47 passed, 6 skipped`.

## Canonical State Impact
Updated `.agents/memory/state/PROJECT_STATE.md` to record that `aria_nbv.data_handling` is the canonical owner of raw snippet, oracle-cache, and VIN-cache contracts, while `aria_nbv.data` is the compatibility surface only.

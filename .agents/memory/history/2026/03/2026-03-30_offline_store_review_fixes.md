---
id: 2026-03-30_offline_store_review_fixes
date: 2026-03-30
title: "Offline Store Review Fixes"
status: done
topics: [data-handling, offline-store, migration, testing]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/data_handling/__init__.py
  - aria_nbv/aria_nbv/data/offline_cache_types.py
  - aria_nbv/aria_nbv/data_handling/_offline_writer.py
  - aria_nbv/aria_nbv/data_handling/_migration.py
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
---

Task: implement the minimal fixes requested from PR review findings around offline split reproducibility, migration overwrite safety, VIN-cache compatibility checks, and the legacy compatibility import path.

Method: patched the offline writer to assign splits by stable `sha1(sample_key)` ranking while preserving in-split order, added manifest provenance for the split policy, changed the legacy shim to import through `aria_nbv.data_handling`, bound the cache-contract types on the package root without adding them to `__all__`, and hardened migration to validate reused VIN-cache settings plus swap destination stores only after a successful temp-store build.

Findings/outputs: the original public API contract failure was resolved by moving the legacy shim to the root contract; migration now fails early on incompatible reused VIN-cache settings; overwrite mode keeps the previous migrated store intact if conversion fails before the final swap.

Verification:
- `aria_nbv/.venv/bin/ruff check aria_nbv/aria_nbv/data_handling/__init__.py aria_nbv/aria_nbv/data/offline_cache_types.py aria_nbv/aria_nbv/data_handling/_offline_writer.py aria_nbv/aria_nbv/data_handling/_migration.py aria_nbv/tests/data_handling/test_vin_offline_store.py`
- `aria_nbv/.venv/bin/ruff format aria_nbv/aria_nbv/data_handling/__init__.py aria_nbv/aria_nbv/data/offline_cache_types.py aria_nbv/aria_nbv/data_handling/_offline_writer.py aria_nbv/aria_nbv/data_handling/_migration.py aria_nbv/tests/data_handling/test_vin_offline_store.py`
- `cd aria_nbv && ../aria_nbv/.venv/bin/python -m pytest -s tests/data_handling/test_public_api_contract.py tests/data_handling/test_vin_offline_store.py`
- `cd aria_nbv && ../aria_nbv/.venv/bin/python -m pytest -s tests/data_handling/test_cache_v2.py`
- `cd aria_nbv && ../aria_nbv/.venv/bin/python -m pytest -s tests/data/test_offline_cache_split.py`

Canonical state impact: none. The fixes tighten implementation behavior and regression coverage but do not change the canonical project narrative or decision records.

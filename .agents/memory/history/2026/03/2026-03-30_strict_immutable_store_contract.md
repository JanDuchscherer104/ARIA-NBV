---
id: 2026-03-30_strict_immutable_store_contract
date: 2026-03-30
title: "Strict Immutable Store Contract"
status: done
topics: [data-handling, offline-store, migration]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - aria_nbv/aria_nbv/data_handling/_config_utils.py
  - aria_nbv/aria_nbv/data_handling/_offline_format.py
  - aria_nbv/aria_nbv/data_handling/_offline_store.py
  - aria_nbv/aria_nbv/data_handling/_sample_keys.py
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
  - aria_nbv/aria_nbv/data_handling/README.md
  - .agents/memory/state/PROJECT_STATE.md
assumptions:
  - Dedicated migration tooling remains the only supported path for bringing legacy oracle/VIN cache data into the immutable store.
---

Task
- Remove the runtime backward-compatibility branch for older immutable-store record blocks and keep only the explicit migration path.

Method
- Dropped the `msgpack_records` runtime branch from the immutable store reader, required exact format version 4 at reader initialization, and made unsupported block kinds fail fast with a rebuild-oriented error.
- Added the shared `_config_utils.py` and `_sample_keys.py` helpers to the tracked tree because committed `data_handling` modules already import them.
- Replaced the legacy-read test with rejection tests for outdated manifest versions and record-block kinds.
- Updated the immutable-store README and project state to describe the stricter contract.

Verification
- `cd aria_nbv && ruff check aria_nbv/data_handling/_offline_format.py aria_nbv/data_handling/_offline_store.py tests/data_handling/test_vin_offline_store.py`
- `cd aria_nbv && uv run pytest --capture=no tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py tests/lightning/test_vin_datamodule_sources.py`
- `make check-agent-memory`

Canonical State Impact
- `PROJECT_STATE.md` now records that version 4 is the only supported immutable-store runtime format and that older stores must be rebuilt through migration tooling.

---
id: 2026-03-30_indexed_msgpack_record_blocks
date: 2026-03-30
title: "Indexed MessagePack Record Blocks For VIN Offline Store"
status: done
topics: [data-handling, offline-store, storage-format]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - aria_nbv/aria_nbv/data_handling/_offline_format.py
  - aria_nbv/aria_nbv/data_handling/_offline_store.py
  - aria_nbv/aria_nbv/data_handling/README.md
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
---

Task
- Remove the remaining whole-shard decode behavior for optional immutable-store diagnostics without breaking existing migrated stores.

Method
- Bumped the immutable-store format to version 4 and changed optional record blocks to write one concatenated MessagePack payload blob plus a NumPy offsets sidecar.
- Kept reader compatibility for older `msgpack_records` manifests so pre-existing stores continue to load.
- Added focused tests for new indexed writes, indexed reads that bypass the legacy list decoder, and legacy list fallback reads.

Verification
- `cd aria_nbv && ruff check aria_nbv/data_handling/_offline_format.py aria_nbv/data_handling/_offline_store.py tests/data_handling/test_vin_offline_store.py`
- `cd aria_nbv && uv run pytest --capture=no tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py tests/lightning/test_vin_datamodule_sources.py`

Canonical State Impact
- Updated `PROJECT_STATE.md` to record the version-4 indexed MessagePack layout and the continued backward-read support for older immutable stores.

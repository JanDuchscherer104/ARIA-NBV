---
id: 2026-03-30_migration_cli_and_offline_runtime_cleanup
date: 2026-03-30
title: "Migration CLI And Offline Runtime Cleanup"
status: done
topics: [data-handling, migration, offline-store, packaging]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - .agents/workspace/data_handling_migration/scan_legacy_offline_data.py
  - .agents/workspace/data_handling_migration/convert_legacy_to_vin_offline.py
  - .agents/workspace/data_handling_migration/verify_migrated_vin_offline.py
  - aria_nbv/pyproject.toml
  - aria_nbv/aria_nbv/data_handling/__init__.py
  - aria_nbv/aria_nbv/data_handling/_vin_sources.py
  - aria_nbv/aria_nbv/data_handling/_legacy_vin_source.py
  - aria_nbv/aria_nbv/data_handling/_offline_dataset.py
  - aria_nbv/aria_nbv/data_handling/_migration.py
  - aria_nbv/aria_nbv/data_handling/README.md
  - aria_nbv/tests/data_handling/test_public_api_contract.py
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
  - aria_nbv/tests/lightning/test_vin_datamodule_sources.py
  - docs/contents/resources/agent_scaffold/.gitignore
---

Task
- Resolve the confirmed post-refactor regressions around the migration workspace CLIs and the missing legacy cache entry point, then tighten the low-risk offline-store runtime issues found during review.

Method
- Switched the three workspace migration CLIs to import legacy cache configs from `aria_nbv.data_handling._legacy_cache_api` while keeping canonical migration entry points on the package root.
- Restored the `nbv-cache-samples` console script while the underlying legacy cache CLI still exists and is still documented.
- Removed the ambiguous `VinOracleDatasetConfig` alias from the canonical package root, kept it on the dedicated legacy wrapper surface, changed the legacy cache source defaults to `train`/`val`, and made immutable-store `vin_batch` reads bypass optional diagnostic record decoding.
- Strengthened `verify_migrated_offline_data()` to compare migrated provenance and core numeric blocks against the legacy oracle/VIN payloads.

Verification
- `ruff format` and `ruff check` on the touched runtime and test files
- `aria_nbv/.venv/bin/python -m ruff check` on the three workspace migration CLIs
- `cd aria_nbv && uv run pytest --capture=no tests/data_handling/test_public_api_contract.py tests/data_handling/test_vin_offline_store.py tests/lightning/test_vin_datamodule_sources.py`
- Command smokes:
  - `aria_nbv/.venv/bin/python .agents/workspace/data_handling_migration/scan_legacy_offline_data.py --help`
  - `aria_nbv/.venv/bin/python .agents/workspace/data_handling_migration/convert_legacy_to_vin_offline.py --help`
  - `aria_nbv/.venv/bin/python .agents/workspace/data_handling_migration/verify_migrated_vin_offline.py --help`
  - `cd aria_nbv && uv run nbv-cache-samples --help`

Canonical State Impact
- Updated `PROJECT_STATE.md` to record the canonical root export cleanup and the stronger migrated-store verification semantics.

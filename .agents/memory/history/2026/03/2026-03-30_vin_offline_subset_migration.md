---
id: 2026-03-30_vin_offline_subset_migration
date: 2026-03-30
title: "VIN Offline Subset Migration"
status: done
topics: [data_handling, migration, offline_store, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - aria_nbv/aria_nbv/data_handling/_migration.py
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
  - aria_nbv/aria_nbv/data_handling/README.md
  - .agents/workspace/data_handling_migration/scan_legacy_offline_data.py
  - .agents/workspace/data_handling_migration/convert_legacy_to_vin_offline.py
  - .agents/workspace/data_handling_migration/verify_migrated_vin_offline.py
  - .agents/workspace/data_handling_migration/run_migration.sh
  - .agents/workspace/data_handling_migration/README.md
  - .agents/memory/state/PROJECT_STATE.md
artifacts:
  - /tmp/vin_offline_subset_smoke
assumptions:
  - The workspace oracle cache at /home/jandu/repos/NBV/.data/oracle_rri_cache remains the authoritative legacy source for subset migration smoke tests.
---

Task: verify whether the immutable VIN offline store had already been materialized on disk, then make the migration path usable for subset-first testing.

Method: inspected the current workspace caches, fixed the legacy scan metadata snapshot in `data_handling._migration`, added subset selectors (`scene_ids`, `split`, `max_records`) to scan/convert/verify, extended the focused offline-store tests, and updated the temporary migration workspace docs and wrapper script.

Findings: there was no committed `.data/vin_offline` store in this workspace, and no legacy `.data/vin_snippet_cache` directory either. The real legacy source is `.data/oracle_rri_cache`. The updated scan CLI now works on that cache, and the convert path can rebuild VIN snippets live from the dataset payload embedded in oracle-cache metadata when no VIN cache is present.

Verification:
- `ruff check` on the touched migration/runtime/test files
- `uv run pytest --capture=no tests/data_handling/test_vin_offline_store.py`
- `.venv/bin/python .agents/workspace/data_handling_migration/scan_legacy_offline_data.py --oracle-cache /home/jandu/repos/NBV/.data/oracle_rri_cache --split train --max-records 2`
- `.venv/bin/python .agents/workspace/data_handling_migration/convert_legacy_to_vin_offline.py --oracle-cache /home/jandu/repos/NBV/.data/oracle_rri_cache --out-store /tmp/vin_offline_subset_smoke --split train --max-records 1 --workers 0 --samples-per-shard 1 --overwrite`
- `.venv/bin/python .agents/workspace/data_handling_migration/verify_migrated_vin_offline.py --oracle-cache /home/jandu/repos/NBV/.data/oracle_rri_cache --split train --max-records 1 --store /tmp/vin_offline_subset_smoke`

Canonical state impact: `PROJECT_STATE.md` now records that subset-first migration is part of the supported operator workflow for the immutable VIN offline store.

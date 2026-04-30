---
id: 2026-04-30_offline_writer_cli_config
date: 2026-04-30
title: "Offline Writer CLI And 81286 Build Config"
status: done
topics: [data-handling, offline-store, oracle-rri, vin]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/pyproject.toml
  - aria_nbv/aria_nbv/data_handling/offline_cli.py
  - aria_nbv/aria_nbv/data_handling/_offline_writer.py
  - aria_nbv/aria_nbv/vin/backbone_evl.py
  - aria_nbv/aria_nbv/vin/types.py
  - .configs/build_vin_offline_81286.toml
---

## Task

Added a first-class immutable VIN offline-store build entrypoint and a full-scene
scene 81286 writer config for oracle-RRI data generation.

## Method

- Added `nbv-build-offline` for loading `VinOfflineWriterConfig` TOML files.
- Added writer keep-lists for numeric backbone blocks and rich backbone payloads.
- Made `EvlBackboneConfig.features_mode` constrain returned feature families.
- Added `.configs/build_vin_offline_81286.toml` targeting the canonical
  `vin_offline` store with rich oracle diagnostics and head-only EVL outputs.

## Verification

- `cd aria_nbv && uv run nbv-build-offline --config-path build_vin_offline_81286.toml --dry-run`
- `cd aria_nbv && uv run ruff check aria_nbv/data_handling/_offline_writer.py aria_nbv/data_handling/offline_cli.py aria_nbv/vin/backbone_evl.py aria_nbv/vin/types.py tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py tests/vin/test_backbone_evl.py`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py tests/vin/test_backbone_evl.py`

## Notes

No full offline store was built in this slice. `overwrite = false` remains the
default in the 81286 config, so an existing canonical store will not be replaced
without an explicit config change.

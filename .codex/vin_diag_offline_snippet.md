# VIN diagnostics offline snippet + navigation

## Summary
- Added offline cache navigation (next sample + index) and optional EFM snippet attachment for VIN diagnostics.
- Implemented cache-backed batch construction using `OracleRriCacheDataset` and an EFM lookup via `AseEfmDatasetConfig`.
- Extended `VinDiagnosticsState` to cache offline dataset state and snippet lookups.

## Key changes
- `oracle_rri/oracle_rri/app/panels.py`: offline cache dataset prep, next-sample UI, cache-index selection, EFM snippet loading and batch construction.
- `oracle_rri/oracle_rri/app/state_types.py`: new fields for offline cache/snippet state.

## Tests
- `ruff format oracle_rri/oracle_rri/app/panels.py oracle_rri/oracle_rri/app/state_types.py`
- `ruff check oracle_rri/oracle_rri/app/panels.py oracle_rri/oracle_rri/app/state_types.py`
  - Fails due to existing lint issues in both files (relative imports, missing docstrings, etc.).

## Notes / Suggestions
- If desired, promote `_vin_oracle_batch_from_cache` to a shared helper in `lit_datamodule.py` to avoid duplication in the UI.
- `attach EFM snippet` uses cache metadata’s `dataset_config` when available; ensure caches are built with correct `atek_variant` so shard resolution hits the right directory.

## Update
- Geometry tab now auto-loads the snippet/mesh on demand if it was not attached during the offline cache run (using cached metadata when available).

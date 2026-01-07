# Data-flow streamlining notes (2026-01-04)

## Changes implemented
- Added shared VIN snippet utilities: `build_vin_snippet_view`, `empty_vin_snippet`, and a stable cache config hash in `oracle_rri/oracle_rri/data/vin_snippet_utils.py`.
- Introduced `EfmSnippetLoader` (`oracle_rri/oracle_rri/data/efm_snippet_loader.py`) to reuse per-scene `AseEfmDataset` instances across cache and snippet-cache paths.
- Added `vin_snippet_provider` abstractions (`oracle_rri/oracle_rri/data/vin_snippet_provider.py`) to unify VIN snippet loading from either a VIN snippet cache or live EFM access.
- Added `OracleRriCacheVinDataset` wrapper to ensure VIN batches are always returned, simplifying the datamodule code path.
- Updated `VinDataModule` to derive train/val cache configs without mutating `config.train_cache` and to respect an explicit `val_cache` even when `train_cache` is unset.

## Findings / rationale
- The previous dual return types (`OracleRriCacheSample` vs `VinOracleBatch`) made the datamodule path brittle. Wrapping in `OracleRriCacheVinDataset` removes branching and enforces consistent VIN batches.
- A provider chain (vin-snippet cache → EFM loader) centralizes the snippet source decision and adds a compatibility check via a hash computed from the offline cache metadata + semidense settings.

## Potential issues / gotchas
- If offline cache metadata lacks `dataset_config`, the VIN snippet cache hash cannot be validated; cache compatibility checks become best-effort. Consider enforcing `dataset_config` presence at cache creation time.
- `vin_snippet_cache_mode="required"` will raise if the cache metadata hash mismatches; this is desired but may surprise users if the cache was built with older configs.
- The VIN snippet cache still depends on the oracle cache split index (`train_index.jsonl` / `val_index.jsonl`). If those are missing or stale, the snippet cache will mirror that state.

## Suggestions for follow-up
- Surface `vin_snippet_cache_mode` and expected hash in CLI/logging to make cache mismatch behavior more transparent.
- Consider adding a small CLI utility to validate VIN snippet cache metadata against an oracle cache without loading samples.

## Tests executed
- `oracle_rri/.venv/bin/python -m pytest -q tests/data/test_vin_snippet_cache_datamodule_equivalence.py`
  - Validated parity between live EFM-derived VIN snippets and cached VIN snippets for both single and batched cases.

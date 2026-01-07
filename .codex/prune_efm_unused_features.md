# Prune unused EFM features

## Summary
- Added an EFM pruning path so VIN batches only carry required keys (pose + semidense metadata by default), reducing DataLoader payload size.
- Added `efm_keep_keys` to `OracleRriCacheDatasetConfig` and prune cached snippets on load.
- Added `efm_keep_keys` + `prune_efm_snippet` to `VinDataModuleConfig`, propagated the allowlist to cache configs and online/cache datasets.

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_offline_cache_split.py tests/data/test_efm_dataset_snippet.py`

## Notes / Suggestions
- If shared-memory errors persist, consider setting `include_efm_snippet=False` when semidense/trajectory features are disabled, or reduce `num_workers`.
- For more aggressive IO reduction, consider adding a dataset-level key-mapping filter in `AseEfmDataset` to avoid loading unused keys from WDS (requires validating EfmModelAdaptor dependencies).

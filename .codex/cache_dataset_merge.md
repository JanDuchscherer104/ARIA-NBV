# OracleRriCacheDataset Vin Batch Merge

## Findings
- `OracleRriCacheDataset` now supports `return_format="vin_batch"` and optional `simplification`, eliminating the `VinOracleCacheDataset` wrapper.
- `VinDataModule` forces cache configs to return VIN batches and uses a simplified cache instance when `use_train_as_val` is enabled.
- `OracleRriCacheDataset` can now return `VinOracleBatch` directly (via a local import) while retaining default cache-sample behavior.

## Potential issues
- `return_format="vin_batch"` introduces a runtime import dependency on `oracle_rri.lightning.lit_datamodule`.
- Simplification applies after train/val split; tiny splits can collapse to zero samples.

## Suggestions
- Consider moving `VinOracleBatch` to a shared types module to avoid data->lightning coupling.
- Add a `min_samples` guard when simplification would drop to zero in small caches.

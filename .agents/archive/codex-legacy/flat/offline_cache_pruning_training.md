# Offline cache pruning for VIN training (2026-01-03)

## Summary
- Added `load_candidates`, `load_depths`, and `load_candidate_pcs` flags to `OracleRriCacheDatasetConfig`.
- `OracleRriCacheDataset.__getitem__` now skips decoding candidates/point clouds when not needed and can build `VinOracleBatch` directly.
- `LitDataModule` forces cache configs used for training/validation (`return_format=vin_batch`) to disable candidate/candidate_pc decoding.
- Added a unit test ensuring candidate decoding is skipped in `vin_batch` mode.

## Potential issues
- `torch.load` still deserializes the whole payload, so large tensors are still read from disk; decoding skips only avoid extra processing/holding references.
- `cache_sample` format still requires candidates/point clouds; turning off those flags will raise.

## Suggestions
- Split cache payload into multiple files (e.g., depths/rri vs. candidate_pcs) to truly avoid loading unused tensors.
- Add an LRU for snippet loaders or shard index to reduce per-snippet scan time.

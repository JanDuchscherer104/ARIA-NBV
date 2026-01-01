# Offline cache default shuffle

## Summary
- Updated `OracleRriCacheWriterConfig` to default to `AseEfmDatasetConfig(wds_shuffle=True)` so offline cache creation shuffles shards by default.
- Added README note to document the new default and how to override for deterministic order.

## Files touched
- `oracle_rri/oracle_rri/data/offline_cache.py`
- `oracle_rri/oracle_rri/data/README.md`

## Notes
- Change affects only the cache writer default; other dataset usage remains unchanged unless explicitly configured.

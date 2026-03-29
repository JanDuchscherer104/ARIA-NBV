# Remove map_location from OracleRriCacheDataset

- Removed `map_location` from `OracleRriCacheDatasetConfig` and forced CPU loading in `OracleRriCacheDataset`.
- Updated call sites and UI panels to stop passing map_location for oracle caches; kept map_location only for VIN snippet caches where still supported.
- Simplified logs and dataset summaries to reflect fixed CPU map_location.

Suggestions:
- Consider adding a brief migration note in docs/config templates if any existing TOML files still include `map_location` for oracle cache configs.

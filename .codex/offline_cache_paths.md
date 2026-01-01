Offline cache path handling

What changed
- Cache directories now resolve relative to `PathConfig.data_root` in `oracle_rri/oracle_rri/data/offline_cache.py`.
- Added `paths: PathConfig` to cache writer/reader/appender configs and to `VinDataModuleConfig` so cache paths
  propagate consistently from top-level configs.
- Updated the offline cache README example to avoid `.data/.data` double-prefixes.

Potential issues / risks
- If a user previously passed relative `.data/...` cache_dir, the new resolver treats that as relative to `data_root`;
  we added a guard for leading `data_root.name`, but custom `data_root` names may still need manual adjustment.
- Appender config now expects paths propagation; make sure nested configs are revalidated after changes to `PathConfig`.

Suggestions / next steps
- If you regularly change `PathConfig.data_root`, prefer absolute cache_dir paths in configs to avoid ambiguity.
- Add a small unit test for `OracleRriCacheConfig._resolve_cache_dir` with custom `data_root` to lock behavior.

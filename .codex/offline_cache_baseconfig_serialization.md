# Offline cache/BaseConfig simplification (2025-12-31)

## Summary of changes
- Centralized cache metadata serialization in BaseConfig via `model_dump_cache`, supporting declarative cache exclusions.
- Added `cache_exclude_fields` to `AseEfmDatasetConfig` to drop `tar_urls` and `scene_to_mesh` from cache metadata.
- Extended BaseConfig JSON conversion to handle `torch.Tensor`.
- Removed `BaseSettings` from CLI/BaseConfig mixin classes to avoid MRO conflicts.
- Fixed offline cache dataclass decoding to resolve postponed annotations via `get_type_hints` with injected globals.

## Key fixes
- `nbv-cache-samples` now runs after removing BaseSettings/BaseConfig MRO conflicts.
- `nbv-summary` now loads cached PoseTW/CameraTW correctly by resolving postponed annotations and providing PoseTW in type hint evaluation.

## Notes / suggestions
- Consider adding `cache_exclude_fields` for any config fields that might hold large tensors (e.g., `view_target_point_world`) if they become common.
- Future: evaluate whether `PathConfig` should be excluded or normalized for portability when sharing cache metadata across machines.
- The real-data pytest integration for offline cache skipped due to environment (likely missing dependency); CLI runs were used as real-data validation.

## PathConfig integration update
- `OracleRriCacheConfig.cache_dir` now defaults to `PathConfig().data_root / "oracle_rri_cache"` and resolves relative paths via `PathConfig.resolve_under_root`, preferring `offline_cache_dir` when present.
- This keeps cache placement consistent with PathConfig and allows users to pass `offline_cache_dir/...` or `.data/...` paths.

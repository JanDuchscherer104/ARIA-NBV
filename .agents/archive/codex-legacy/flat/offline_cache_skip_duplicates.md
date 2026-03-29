# Offline cache duplicate skipping (2026-01-03)

## Summary
- Added duplicate detection in `OracleRriCacheWriter.run` to skip samples that already exist by `(scene_id, snippet_id)`.
- Preserved existing index entries (when `overwrite=True`) and updated metadata to track the total sample count.
- Added unit test for skip behavior and extended the real-data roundtrip test to verify re-run skipping.

## Potential issues
- `overwrite=True` now preserves existing entries; to fully reset a cache, remove the cache directory manually or rebuild the index from scratch.
- If `index.jsonl` is missing but sample files exist, duplicates may still be generated unless `rebuild_cache_index(...)` is called first.
- Duplicate skipping is based on `(scene_id, snippet_id)` only, so regenerating the same sample with a different config hash will be skipped.

## Suggestions
- Add an explicit `skip_existing` / `preserve_index` toggle and CLI flag to control duplicate handling.
- Warn when existing cache metadata/config hash differs from the current run before reusing entries.
- Short-circuit dataset iteration entirely when `max_samples` is already satisfied by existing entries.

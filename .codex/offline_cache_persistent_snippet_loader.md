# Offline cache persistent snippet loader (2026-01-03)

## Summary
- Added a per-worker `_EfmSnippetLoader` inside `OracleRriCacheDataset` that keeps `AseEfmDataset` instances alive per scene.
- `OracleRriCacheDataset.__getitem__` now reuses these datasets when `include_efm_snippet=True` instead of re-instantiating per sample.

## Potential issues
- Each worker keeps one `AseEfmDataset` per visited scene; memory could grow if many scenes are sampled.
- Snippet lookup still iterates over scene shards; it avoids dataset construction but not shard scanning.

## Suggestions
- Add an LRU cap for `_EfmSnippetLoader._datasets` to bound memory usage.
- Consider a shard-level or snippet-level index to avoid scanning full scene shards for each snippet.

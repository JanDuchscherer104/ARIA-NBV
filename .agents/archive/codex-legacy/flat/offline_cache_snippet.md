# Offline Cache Snippet Loading + Random Split Rebuild

## Findings
- `OracleRriCacheDataset` can now optionally attach `EfmSnippetView` per cached sample (`include_efm_snippet`) and request GT mesh loading (`include_gt_mesh`).
- VIN diagnostics prefer cache-provided snippets and only fall back to manual loading; UI now exposes a GT-mesh toggle.
- `rebuild_cache_index` now re-creates `train_index.jsonl`/`val_index.jsonl` using a randomized split driven by `train_val_split` (seeded when provided).

## Potential issues
- Snippet loading per cached sample is expensive; keep `include_efm_snippet=False` for training loops unless explicitly needed.
- `include_gt_mesh=True` does not guarantee a mesh is present; missing meshes are warned but not fatal.

## Suggestions
- Add a lightweight in-memory snippet cache in `OracleRriCacheDataset` for repeated access in diagnostics.
- Consider persisting the split seed in cache metadata for traceability.

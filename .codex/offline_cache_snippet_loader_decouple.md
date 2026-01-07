# Offline cache snippet loading decouple (2026-01-03)

## Summary
- Decoupled Streamlit cache datasets from `include_efm_snippet` / `include_gt_mesh` so toggling snippet attach no longer re-instantiates `OracleRriCacheDataset`.
- Added on-demand snippet loading for the attribution panel (mirrors VIN diagnostics behavior).
- Cache dataset config signature no longer changes with attach-snippet toggles in the UI.

## Potential issues
- On-demand snippet loading still creates a fresh `AseEfmDataset` per fetch; repeated loads could be expensive without a shared loader cache.
- Attribution panel warns per-sample if snippet loading fails; repeated warnings could be noisy for long runs.

## Suggestions
- Introduce a shared snippet loader with a small LRU cache (scene+snippet keyed) to reuse dataset instances.
- Consider adding a UI flag for `include_gt_mesh` in attribution if mesh-aware features are needed.
- Optionally move `_load_efm_snippet_for_cache` to a shared module to remove duplicate definitions.

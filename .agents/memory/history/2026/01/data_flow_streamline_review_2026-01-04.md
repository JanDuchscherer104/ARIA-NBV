---
id: 2026-01-04_data_flow_streamline_review_2026-01-04
date: 2026-01-04
title: "Data Flow Streamline Review 2026 01 04"
status: legacy-imported
topics: [data, flow, streamline, 2026, 01]
source_legacy_path: ".codex/data_flow_streamline_review_2026-01-04.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Data Flow Review (lit_datamodule / offline_cache / vin_snippet_cache)

Date: 2026-01-04

## Key Findings
- **Dual snippet construction paths**: `OracleRriCacheDataset._build_vin_snippet` and `vin_snippet_cache._build_vin_snippet` duplicate logic with slightly different knobs (`collapse_semidense`, `semidense_max_points`, `include_inv_dist_std`). This risks subtle divergence between cache-built vs on-the-fly snippets.
- **Redundant snippet loaders**: `_EfmSnippetLoader` (offline_cache) vs `_SnippetLoader` (vin_snippet_cache) are near-identical. Harder to ensure consistent dataset config handling.
- **Mixed return types**: `OracleRriCacheDataset` supports both `OracleRriCacheSample` and `VinOracleBatch` via `return_format`, plus `VinOracleBatch.efm_snippet_view` can be `EfmSnippetView` or `VinSnippetView`. This pushes complexity into `lit_datamodule` (collation constraints) and `lit_module` (branching behavior).
- **Config mutation during setup**: `VinDataModule.setup` mutates `train_cache`/`val_cache` fields (return_format, load_candidates, etc.). This makes actual runtime behavior opaque and makes configs harder to reason about.
- **Cache signature mismatch risk**: `VinSnippetCache` metadata/config hash is not validated against training-time `semidense_max_points`/`include_inv_dist_std`, so stale or mismatched snippet caches can silently be used.
- **Implicit fallback**: `OracleRriCacheDataset` silently falls back to EFM snippet loading when `vin_snippet_cache` is missing, which can mask misconfiguration or cache staleness.

## Streamlining Proposal (Incremental)
1. **Single VinSnippetView builder**
   - Move snippet construction to a shared helper (e.g., `oracle_rri/data/vin_snippet_utils.py`).
   - Use it from both offline cache and vin snippet cache builder to ensure identical behavior.

2. **Introduce Snippet Provider interface**
   - `VinSnippetProvider.get(scene_id, snippet_id, map_location) -> VinSnippetView | None`.
   - Implementations: `FromEfmSnippet`, `FromVinSnippetCache`.
   - Inject provider into the offline cache dataset instead of embedding cache logic inside it.

3. **Split datasets by purpose**
   - `OracleRriCacheVinDataset`: always returns `VinOracleBatch` (training/eval path).
   - `OracleRriCacheReader`: returns `OracleRriCacheSample` (analysis/debug path).
   - Remove `return_format` from `OracleRriCacheDatasetConfig` for the training dataset.

4. **Narrow batch type**
   - For offline training, always populate `VinOracleBatch.efm_snippet_view` with `VinSnippetView`.
   - Keep `EfmSnippetView` usage internal to online labeler or analysis paths only.

5. **Config simplification**
   - Add a small factory on `VinDataModuleConfig` (e.g., `build_offline_cache_cfg()`), avoiding in-place mutation.
   - Move shared snippet knobs (`semidense_max_points`, `include_inv_dist_std`) to a single config source and propagate to both cache reader and snippet cache writer.

6. **Strict cache compatibility checks**
   - Compare snippet cache metadata hash with expected config; warn or raise if mismatched.
   - Add `vin_snippet_cache_mode: Literal['required', 'auto', 'disabled']` to remove silent fallback ambiguity.

## Suggested Next Steps
- Decide on a target end-state: *single dataset for training* + separate *reader for analysis*.
- Implement the shared `VinSnippetView` builder and reuse in both paths.
- Add provider + strict cache validation.
- Update tests to cover “required” mode and mismatch detection.

## Tests Already Added
- `tests/data/test_vin_snippet_cache_datamodule_equivalence.py` verifies datamodule outputs match between live EFM snippet loading and VinSnippetCache paths (single + batched). Reuse it after refactors to guard against regressions.

# Offline Oracle Cache (EVL + OracleRRI)

## Changes
- Added `oracle_rri/data/offline_cache.py` with cache writer/reader, config snapshots (excluding `tar_urls`/`scene_to_mesh` but recording counts), and serialization helpers for `CandidateSamplingResult`, `CandidateDepths`, `CandidatePointClouds`, `RriResult`, and `EvlBackboneOutput`.
- Extended `EvlBackboneOutput` to allow optional tensors and added `.to(device)` convenience.
- Integrated cached datasets into `VinDataModule` and `VinLightningModule` (optional `backbone_out`, camera device moves, cache-aware batch construction).
- Added integration test `tests/data/test_offline_cache.py` (skips if `power_spherical` or required real data are missing).
- Updated `oracle_rri/data/README.md` and `docs/contents/todos.qmd` to document/cache the new offline dataset path.

## Notable Findings
- Pytest collection previously failed when `power_spherical` was missing because `oracle_rri` imports it transitively; the test now skips before importing `oracle_rri` modules.

## Suggestions / Follow-ups
- Consider adding a CLI entry point to build caches (e.g., `nbv-cache-oracle`) for easier reproducibility.
- Optional: add skip/retry logic in `OracleRriCacheWriter` to gracefully handle samples with zero candidates (mirrors `VinOracleIterableDataset`).
- Run the new integration test with real data + `power_spherical` installed to validate the full cache round-trip.

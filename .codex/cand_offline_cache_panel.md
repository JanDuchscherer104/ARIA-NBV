# Candidate Panel Offline Cache Support (2026-01-26)

## Goal
Enable the candidate poses panel to visualize cached candidates from `OracleRriCacheDataset` without requiring the online oracle pipeline.

## Approach
- Added a sidebar source switch to choose between online (oracle labeler) and offline cache.
- Reuse cached `CandidateSamplingResult` and (optionally) cached `EfmSnippetView` for plotting.
- If a snippet is not attached, fall back to lightweight plots (candidate centers + frusta) without mesh/scene context.
- Reconstruct `CandidateViewGeneratorConfig` from `metadata.labeler_config.generator` to label plots.

## Changes
- New helpers in `oracle_rri/oracle_rri/app/panels/candidates.py`:
  - Offline cache loading + caching in `st.session_state`.
  - Minimal plotting path for cached samples without EFM snippet.
- UI text clarifies when cached settings are used.

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_efm_dataset_snippet.py`

## Follow-ups
- Consider a dedicated cache sample selector shared with VIN diagnostics to avoid duplicated cache state.
- Optional: show cached labeler config summary (e.g., sampling radii/angles) in the UI.

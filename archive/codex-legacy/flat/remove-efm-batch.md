# Task: Remove redundant efm from VinOracleBatch

## Summary
- Removed the redundant `efm` field from `VinOracleBatch` and now rely on `efm_snippet_view.efm` when raw EFM inputs are needed.
- Updated VIN Lightning module, VIN v2 summary, and Streamlit panels to pull EFM from the snippet view and to error clearly when only cached data is available.
- Cache-backed batches continue to rely on `backbone_out` without requiring raw EFM inputs.

## Findings / Potential Issues
- Cached oracle batches (`efm_snippet_view=None`) still require `backbone_out` for training/inference; attempts to summarize or debug without raw EFM now raise explicit errors.

## Suggestions
- If cached batches are expected to support summaries/debugging, consider persisting minimal EFM keys in the cache or add a lightweight EFM proxy in `VinOracleBatch`.

# Summary Simplification

## Summary
- Reduced duplication in `summarize` by consolidating tensor extraction logic.

## Changes
- `oracle_rri/oracle_rri/utils/summary.py`: added `_extract_tensor` helper and streamlined `summarize` flow.

## Notes / Suggestions
- If more tensor-like wrappers are added, extend `_extract_tensor` to keep `summarize` concise.

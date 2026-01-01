# nbv-summary Empty Batch Fix

## Summary
- Guarded VIN summary to raise a clear error when no oracle batches are produced.

## Changes
- `oracle_rri/oracle_rri/lightning/aria_nbv_experiment.py`: handle empty iterator in `summarize_vin`.

## Notes / Suggestions
- If this is a cache-path issue, confirm `.data/oracle_rri_cache/index.jsonl` and metadata exist and match the chosen stage.

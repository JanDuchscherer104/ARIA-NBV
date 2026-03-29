# Offline Stats VIN Cache Parity + Conflict Fix

## Summary
- Resolved merge conflict markers in `offline_stats.py` and restored VIN snippet cache stats UI.
- VIN snippet cache stats now aggregate **global** per-point `inv_dist_std` / `obs_count` distributions; per-snippet inv/obs summaries removed.
- Added VIN snippet metrics/plots that are compatible with cached snippet data (global point stats + distributions, point count/trajectory length hists, per-snippet counts table).

## Files Touched
- `oracle_rri/oracle_rri/app/panels/offline_cache_utils.py`
- `oracle_rri/oracle_rri/app/panels/offline_stats.py`

## Tests Run
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache_real_data.py`

## Notes / Suggestions
- If global per-point distributions become memory-heavy, consider optional streaming quantile sketches while still caching raw values when requested.

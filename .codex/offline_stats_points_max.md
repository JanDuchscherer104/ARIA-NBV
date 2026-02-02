# Offline Stats Points Max

## Summary
- Added global `points_max` to VIN snippet cache summary stats.
- Updated Offline Stats UI to display `points_max` alongside other VIN snippet metrics.

## Files Touched
- `oracle_rri/oracle_rri/app/panels/offline_cache_utils.py`
- `oracle_rri/oracle_rri/app/panels/offline_stats.py`

## Tests Run
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache_real_data.py`

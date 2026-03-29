# VIN Snippet Global Stats

## Summary
- Removed per-snippet `inv_dist_std`/`obs_count` summaries; now collect all per-point values and compute global stats after scanning the cache.
- Updated Offline Stats UI to report global point-level metrics and histograms derived from the full per-point distributions.

## Files Touched
- `oracle_rri/oracle_rri/app/panels/offline_cache_utils.py`
- `oracle_rri/oracle_rri/app/panels/offline_stats.py`

## Tests Run
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache_real_data.py`

## Notes / Suggestions
- Global distributions may be large; if memory becomes an issue, consider chunked quantile estimation or a streaming quantile sketch while still preserving full samples in an optional on-disk cache.

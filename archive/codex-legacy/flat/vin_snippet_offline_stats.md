# Vin Snippet Offline Stats: obs_count + inv_dist_std

## Summary
- Added per-point summary stats for `inv_dist_std` and `obs_count` in VIN snippet cache stats collection (mean/std/min/max/p50/p95).
- Surfaced new summary metrics and histograms in the Offline Stats panel for VIN snippet caches.
- Documented Offline Stats page now reporting per-point `inv_dist_std`/`obs_count` mean, std, and p95.

## Files Touched
- `oracle_rri/oracle_rri/app/panels/offline_cache_utils.py`
- `oracle_rri/oracle_rri/app/panels/offline_stats.py`
- `docs/contents/impl/data_pipeline_overview.qmd`

## Tests Run
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache.py`
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache_real_data.py`
- `oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_v2_point_encoder_real_data.py` (FAILED: ImportError for `VinModelV2Config` in `oracle_rri.vin`)

## Notes / Suggestions
- The VIN v2 real-data test currently fails due to `VinModelV2Config` import not exposed in `oracle_rri.vin.__init__`; consider fixing exports or adjusting the test import path.
- If per-snippet normalization needs to be consumed by training, consider persisting `obs_count_max` and `inv_dist_std_p95` in the VIN snippet cache payload for zero-cost reuse.

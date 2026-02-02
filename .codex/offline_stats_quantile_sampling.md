# Offline Stats Quantile Sampling Fix

## Summary
- Fixed `RuntimeError: quantile() input tensor is too large` by sampling large tensors before computing p50/p95.
- Quantiles now computed on a fixed-size random subset (default 200k) with deterministic seed.

## Files Touched
- `oracle_rri/oracle_rri/app/panels/offline_cache_utils.py`

## Tests Run
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache_real_data.py`

## Notes
- Mean/std/min/max still computed on full data; only quantiles are sampled to avoid torch limits.

# Offline Stats cache_dir strip fix

## Issue
- Streamlit Offline Stats crashed with `AttributeError: 'PosixPath' object has no attribute 'strip'` when building cache keys.

## Fix
- Normalize `cache_dir` (and `toml_path`) to string via `_as_path_str` before `.strip()` and key construction.

## Files
- `oracle_rri/oracle_rri/app/panels/offline_stats.py`

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests/data/test_vin_snippet_cache_real_data.py`

## Notes
- UI behavior unchanged besides avoiding crash; `cache_dir` remains a `Path` for downstream path operations.

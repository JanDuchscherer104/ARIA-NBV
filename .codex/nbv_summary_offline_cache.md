# nbv-summary offline cache preference (2025-12-30)

## Summary
- `AriaNBVExperimentConfig.summarize_vin()` now auto-detects the default oracle cache (`.data/oracle_rri_cache`) and uses it for summary runs when it includes backbone + depths + pointclouds.
- Falls back to online oracle labeler when the cache is missing or incomplete, logging the reason.
- Added VIN package wrappers for `coral` and `rri_binning` to mirror the new `rri_metrics` location, and exported `RriOrdinalBinner`/CORAL utilities from `rri_metrics.__init__` so CLI imports resolve.

## Files touched
- `oracle_rri/oracle_rri/lightning/aria_nbv_experiment.py`
- `oracle_rri/oracle_rri/vin/coral.py`
- `oracle_rri/oracle_rri/vin/rri_binning.py`
- `oracle_rri/oracle_rri/rri_metrics/__init__.py`

## Command run
- `uv run nbv-summary` (from `oracle_rri/`) completed successfully. It reported no cache found on this machine and used the online oracle labeler.

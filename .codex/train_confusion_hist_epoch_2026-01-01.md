# Train confusion/hist epoch accumulation (2026-01-01)

## Summary
- Split interval metrics from epoch metrics so train confusion matrix + histogram can accumulate over the full epoch.
- Added `_interval_metrics` for step logging; epoch metrics now remain intact for end-of-epoch logging.

## Files touched
- `oracle_rri/oracle_rri/lightning/lit_module.py`

## Tests
- `python -m py_compile oracle_rri/oracle_rri/lightning/lit_module.py`

## Notes
- Interval metrics reset each interval and again at `on_train_epoch_end` to avoid cross-epoch leakage.

## Update: Allow log_interval_steps=None
- Renamed config field to `log_interval_steps` (fixed typo) and added validator to allow `None`.
- `_log_interval_metrics` now returns early when `log_interval_steps` is `None`.

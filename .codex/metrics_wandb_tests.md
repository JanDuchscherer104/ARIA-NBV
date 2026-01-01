# Metrics + W&B Tests

## Summary
- Added unit tests for metrics utilities and W&B logging integration.

## Tests
- `pytest oracle_rri/tests/rri_metrics/test_logging_metrics.py oracle_rri/tests/lightning/test_wandb_logging.py`

## Results
- All tests passed (6 total). Warnings about torchmetrics buffering and deprecation noted.

## Notes / Suggestions
- Tests use lightweight stubs for optional deps (`coral_pytorch`, `power_spherical`, `e3nn`) to avoid import-time failures.

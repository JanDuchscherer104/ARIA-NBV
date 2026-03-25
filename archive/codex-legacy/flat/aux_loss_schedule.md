# Aux loss exponential schedule

## Summary
- Added configurable exponential decay for the auxiliary regression loss weight in `oracle_rri/oracle_rri/lightning/lit_module.py`.
- Introduced config fields for initial weight, decay gamma, minimum weight, and decay interval.
- Logged the current aux weight for monitoring and updated CORAL integration docs with the schedule.
  - Added short theoretical motivation for exponential decay and nonzero floor.

## Notes / Suggestions
- Start with `aux_regression_weight` high (e.g., 5–10) and decay to a small nonzero floor (e.g., 0.1–0.5).
- Use `aux_regression_weight_interval = "epoch"` to keep decay stable across variable batch counts.

# Loss logging namespaces + Loss enum (2026-01-03)

## Summary
- Added a `Loss` `StrEnum` and `loss_key(...)` helper to centralize loss naming and namespaces.
- Routed CORAL diagnostic losses (`coral_loss_balanced_bce`, `coral_loss_focal`) to the `*-aux/` namespace; kept core losses in `stage/`.
- Extended `Metric` with `aux_regression_weight` and `coral_monotonicity_violation_rate` for enum-based logging.

## Key changes
- `oracle_rri/oracle_rri/rri_metrics/logging.py`: new `Loss` enum + `loss_key`, namespace-aware `metric_key`, extra Metric entries.
- `oracle_rri/oracle_rri/lightning/lit_module.py`: log loss keys via `Loss` and move focal/balanced_bce to `stage-aux/`; log aux regression weight/monotonicity via `Metric`.
- `oracle_rri/tests/rri_metrics/test_logging_metrics.py`: added coverage for `loss_key` and aux namespace.

## Findings / issues
- Integration test `oracle_rri/tests/integration/test_vin_lightning_real_data.py` fails before training because `OracleRRIConfig` rejects `candidate_chunk_size` (extra field). This appears unrelated to the logging changes and likely indicates config/test drift.

## Suggestions
- Decide whether `candidate_chunk_size` should be reinstated in `OracleRRIConfig` or removed from the integration test config.
- If desired, move `coral_loss_coral` to `stage-aux/` for consistency (currently still logged in `stage/`).

## Tests
- `ruff format` + `ruff check` (logging/lit_module/tests) OK.
- `pytest oracle_rri/tests/rri_metrics/test_logging_metrics.py` OK (via `.venv`).
- Integration test failed: `pytest oracle_rri/tests/integration/test_vin_lightning_real_data.py` -> `OracleRRIConfig` extra field `candidate_chunk_size`.

## Follow-up (coral_loss_coral cleanup)
- Removed the redundant `coral_loss_coral` logging and enum entry; `coral_loss` is now the single canonical CORAL loss key.
- Moved `aux_regression_loss` into the `*-aux/` namespace alongside focal/balanced_bce diagnostics.

## Follow-up (val bias/variance metrics)
- Added `RriErrorStats` accumulator to compute bias^2 and variance of `pred_rri_proxy - rri` over validation candidates.
- Logged new `val-aux/pred_rri_bias2` and `val-aux/pred_rri_variance` per epoch.
- Added a unit test for the bias/variance calculator.
- Integration test still fails due to `OracleRRIConfig` rejecting `candidate_chunk_size` (pre-existing config/test drift).

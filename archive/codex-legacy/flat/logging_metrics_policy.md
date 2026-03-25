Task: Centralize logging policy for Metric/Loss enums.

Changes:
- Added LogSpec and per-variant log_spec() definitions in `oracle_rri/oracle_rri/rri_metrics/logging.py`.
- Updated `oracle_rri/oracle_rri/lightning/lit_module.py` to group log payloads by LogSpec.
- Added tests in `tests/rri_metrics/test_logging_specs.py`.

Notes:
- Progress bar now only shows Loss.LOSS and Loss.CORAL_REL_RANDOM (train step+epoch, val epoch).
- Other metrics log on epoch only, or step-only for interval metrics (e.g. SPEARMAN_STEP).

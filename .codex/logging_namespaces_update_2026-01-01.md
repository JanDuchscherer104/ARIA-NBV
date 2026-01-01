Title: Logging Namespace Updates (2026-01-01)

Changes
- Moved grad norm logging to `train-gradnorms/*`.
- Loss metrics now live under `train/*` only:
  - `train/loss` (combined), `train/coral_loss`, `train/aux_regression_loss` (if enabled).
- Non-loss scalars now under `{stage}-aux/*` (e.g., `train-aux/rri_mean`, `train-aux/pred_rri_mean`, `train-aux/spearman`).
- Confusion matrix + label histogram figures now under `{stage}-figures/*`.

Files
- `oracle_rri/lightning/lit_module.py`

Tests
- `ruff format oracle_rri/lightning/lit_module.py`
- `ruff check oracle_rri/lightning/lit_module.py`

Notes
- Existing dashboards will need to refresh metric queries for the new namespaces.

# VINv3: Semidense CNN Diagnostics + Toggle (2026-01-26)

## Change
- Added plotting utilities to visualize the semidense CNN grid inputs (occupancy + depth mean/std).
- Integrated new plots into VIN diagnostics tokens tab when `semidense_cnn_enabled` is True.
- Confirmed `semidense_cnn_enabled` config flag gates the CNN in `VinModelV3` (ablation-ready).

## Files
- `oracle_rri/oracle_rri/vin/plotting.py`
- `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/tokens.py`

## Tests
- `ruff format oracle_rri/oracle_rri/vin/plotting.py oracle_rri/oracle_rri/app/panels/vin_diag_tabs/tokens.py`
- `ruff check oracle_rri/oracle_rri/vin/plotting.py oracle_rri/oracle_rri/app/panels/vin_diag_tabs/tokens.py`

# VIN diagnostics checkpoint selector removal

## Summary
- Removed the "Use checkpoint" UI from VIN diagnostics.
- Diagnostics now rely on checkpoint settings inside the selected TOML config.

## Files touched
- `oracle_rri/oracle_rri/app/panels/vin_diagnostics.py`

## Tests
- `ruff check oracle_rri/oracle_rri/app/panels/vin_diagnostics.py`

## Notes
- If a checkpoint is required, set `ckpt_path` in the experiment TOML used by the diagnostics panel.

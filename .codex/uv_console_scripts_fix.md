# uv console scripts fix (2025-12-30)

## Summary
- Replaced invalid console script entries with wrapper functions in `oracle_rri.lightning.cli`.
- Added `train_main()` and `summarize_main()` helpers to force run modes via CLI args.

## Files touched
- `oracle_rri/oracle_rri/lightning/cli.py`
- `oracle_rri/pyproject.toml`

## Notes
- The invalid entry point error came from `[project.scripts]` entries containing inline args.
- `nbv-summary` and `nbv-train` now point to the wrapper functions.

## Tests
- `python -m pytest oracle_rri/tests/integration/test_vin_lightning_real_data.py` (via `oracle_rri/.venv/bin/python`) failed with a pydantic validation error for `ReduceLrOnPlateauConfig.target` (NoTarget not subclass). This appears unrelated to the console script fix.

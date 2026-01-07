# RRI binner utils migration (2026-01-02)

## Summary
- Moved RRI fit-data and binner-loading helpers into `RriOrdinalBinner`.
- Removed `oracle_rri/oracle_rri/app/panels/rri_utils.py` and updated Streamlit panel to call `RriOrdinalBinner.load_fit_data` / `RriOrdinalBinner.load`.

## Findings
- `uv run pytest` used the system Python (3.12) and failed due to missing `power_spherical`; the venv at `oracle_rri/.venv` has the dependency.
- Integration test `oracle_rri/tests/data_handling/test_real_data_integration.py` fails with a Pydantic validation error: `AseEfmDatasetConfig` no longer accepts `verbose`.

## Tests
- `uv run ruff format oracle_rri/oracle_rri/rri_metrics/rri_binning.py oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `uv run ruff check oracle_rri/oracle_rri/rri_metrics/rri_binning.py oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest tests/vin/test_rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/data_handling/test_real_data_integration.py` (failed: `verbose` extra field)

## Suggestions
- Fix or update `AseEfmDatasetConfig` usage in `oracle_rri/tests/data_handling/test_real_data_integration.py` (remove `verbose` or reintroduce field) so integration tests can run.
- Consider standardizing test commands to always use `oracle_rri/.venv/bin/python` to avoid missing dependency errors.

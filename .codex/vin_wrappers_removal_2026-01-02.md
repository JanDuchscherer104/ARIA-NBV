# VIN wrapper removal (2026-01-02)

## Summary
- Removed VIN wrapper modules `oracle_rri/oracle_rri/vin/coral.py` and `oracle_rri/oracle_rri/vin/rri_binning.py`.
- Updated VIN code and tests to import directly from `oracle_rri/rri_metrics/coral.py` and `oracle_rri/rri_metrics/rri_binning.py`.
- Dropped coral/binner re-export entries from `oracle_rri/oracle_rri/vin/__init__.py`.

## Files touched
- `oracle_rri/oracle_rri/vin/model_v1_SH.py` (direct coral import)
- `oracle_rri/oracle_rri/vin/__init__.py` (removed re-export entries)
- `oracle_rri/tests/vin/test_rri_binning.py` (direct binner import)
- `oracle_rri/tests/integration/test_vin_lightning_real_data.py` (direct binner import)
- Deleted: `oracle_rri/oracle_rri/vin/coral.py`, `oracle_rri/oracle_rri/vin/rri_binning.py`

## Tests
- `uv run ruff format oracle_rri/oracle_rri/vin/model_v1_SH.py oracle_rri/oracle_rri/vin/__init__.py oracle_rri/tests/vin/test_rri_binning.py oracle_rri/tests/integration/test_vin_lightning_real_data.py`
- `uv run ruff check oracle_rri/oracle_rri/vin/model_v1_SH.py oracle_rri/oracle_rri/vin/__init__.py oracle_rri/tests/vin/test_rri_binning.py oracle_rri/tests/integration/test_vin_lightning_real_data.py --ignore N999`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest tests/vin/test_rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/integration/test_vin_lightning_real_data.py` (failed; see below)

## Findings
- Integration test failed due to Pydantic validation error: `OracleRRIConfig` no longer accepts `candidate_chunk_size` (extra field).

## Suggestions
- Update `oracle_rri/tests/integration/test_vin_lightning_real_data.py` to match the current `OracleRRIConfig` schema (remove or replace `candidate_chunk_size`) so the integration test can run.

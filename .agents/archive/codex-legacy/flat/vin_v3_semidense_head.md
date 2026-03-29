# VINv3: Semidense Projection Features in Head (2026-01-26)

## Change
- Added `semidense_proj` to the scorer head input in `oracle_rri/oracle_rri/vin/model_v3.py`.
- Updated head input dimension to include `SEMIDENSE_PROJ_DIM`.

## Rationale
- v3 computed semidense projection stats but did not pass them to the scorer, removing candidate-specific cues and contributing to mode-collapse.

## Files
- `oracle_rri/oracle_rri/vin/model_v3.py`

## Tests
- `ruff format oracle_rri/oracle_rri/vin/model_v3.py`
- `ruff check oracle_rri/oracle_rri/vin/model_v3.py`
- `uv run pytest oracle_rri/tests/vin/test_vin_model_v3_core.py oracle_rri/tests/vin/test_vin_model_v3_methods.py`

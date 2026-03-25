# VINv3: Higher Semidense Grid + Tiny CNN (2026-01-26)

## Change
- Increased `semidense_proj_grid_size` default to 24.
- Added a tiny 2D CNN over semidense projection grids (occupancy + depth mean/std).
- Appended CNN output to the scorer head; updated head input dimension accordingly.
- Added grid feature encoding method and tests.
- Added `semidense_grid_feat` to diagnostics.

## Rationale
- Provide richer candidate-specific cues (approximate VIN-NBV grid encoder) while keeping v3 lightweight.

## Files
- `oracle_rri/oracle_rri/vin/model_v3.py`
- `oracle_rri/oracle_rri/vin/types.py`
- `oracle_rri/tests/vin/test_vin_model_v3_methods.py`
- `docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ`
- `.codex/AGENTS_INTERNAL_DB.md`

## Tests
- `ruff format oracle_rri/oracle_rri/vin/model_v3.py oracle_rri/oracle_rri/vin/types.py oracle_rri/tests/vin/test_vin_model_v3_methods.py`
- `ruff check oracle_rri/oracle_rri/vin/model_v3.py oracle_rri/oracle_rri/vin/types.py oracle_rri/tests/vin/test_vin_model_v3_methods.py`
- `uv run pytest oracle_rri/tests/vin/test_vin_model_v3_core.py oracle_rri/tests/vin/test_vin_model_v3_methods.py`
- `typst compile --root docs docs/typst/paper/main.typ`

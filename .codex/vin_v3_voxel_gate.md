# VINv3: Remove Voxel Gate (2026-01-26)

## Change
- Removed voxel gating path in `oracle_rri/oracle_rri/vin/model_v3.py`.
- Enforced `use_voxel_valid_frac_gate=False`; now raises if True.
- Kept voxel projection FiLM as the single modulation path.
- Updated tests to disable the gate.

## Rationale
- Avoid dual modulation (sigmoid gate + FiLM) and align with sweep evidence that gating hurt early learning.

## Files
- `oracle_rri/oracle_rri/vin/model_v3.py`
- `oracle_rri/tests/vin/test_vin_model_v3_methods.py`

## Tests
- `ruff format oracle_rri/oracle_rri/vin/model_v3.py oracle_rri/tests/vin/test_vin_model_v3_methods.py`
- `ruff check oracle_rri/oracle_rri/vin/model_v3.py oracle_rri/tests/vin/test_vin_model_v3_methods.py`
- `uv run pytest oracle_rri/tests/vin/test_vin_model_v3_core.py oracle_rri/tests/vin/test_vin_model_v3_methods.py`

# Optimizable fields fix (2026-01-01)

## Summary
- Updated `optimizable_field` to support `default_factory` (enforces exactly one of `default` or `default_factory`).
- Converted `VinModelV2Config.scene_field_channels` to use `optimizable_field` so the optimizable metadata is attached correctly.

## Files touched
- `oracle_rri/oracle_rri/utils/optuna_optimizable.py`
- `oracle_rri/oracle_rri/vin/model_v2.py`

## Tests
- `python -m py_compile oracle_rri/oracle_rri/utils/optuna_optimizable.py oracle_rri/oracle_rri/vin/model_v2.py`

## Notes / Suggestions
- Any future list-like optimizable fields should use `optimizable_field(default_factory=...)` to avoid shared mutable defaults.

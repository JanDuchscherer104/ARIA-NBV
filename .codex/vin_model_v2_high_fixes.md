# vin_model_v2 high findings fix (2025-12-30)

## Summary
- Fixed pose scaling direction in VIN v2 to match v1 semantics (multiply by learned scales).
- Replaced binary valid_frac with a coverage proxy sampled from `counts_norm` at candidate centers; candidate_valid now requires finite pose + in-bounds center.

## Files touched
- `oracle_rri/oracle_rri/vin/model_v2.py`

## Tests
- `python -m pytest tests/vin/test_vin_model_v2_integration.py` (via `oracle_rri/.venv/bin/python`)

## Notes
- Coverage proxy uses `_sample_voxel_field` on `field_in` with K=1 to sample `counts_norm` at each candidate center.
- `valid_frac` is now continuous in [0,1] and aligns with loss weighting in `VinLightningModule`.

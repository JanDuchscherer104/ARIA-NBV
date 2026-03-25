# Shuffle candidates in VIN batches

## Summary
- Added per-sample candidate shuffling for VIN batches to reduce ordering bias.
- Implemented in the data pipeline (VinOracleBatch + VinDataModule), not the Lightning module.

## Decisions
- VinOracleBatch.shuffle_candidates permutes candidate poses, labels, and PyTorch3D cameras consistently.
- VinDataModule wraps map-style (offline cache) datasets when `shuffle_candidates=true` so shuffling happens before collation/padding.
- Online iterable datasets are left unchanged.

## Tests
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_vin_model_v3_methods.py`

## Follow-ups
- Consider whether online oracle datasets should optionally support candidate shuffling as well.

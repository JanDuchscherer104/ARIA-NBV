# VIN diagnostics OBB batching fix

Date: 2026-01-26

## Issue
VIN diagnostics failed when batching offline cache samples because
`VinOracleBatch.collate` does not support stacking OBB outputs
(`obbs_pr_nms`, `obb_pred`, etc.).

## Fix
- Added helpers to detect and strip OBB outputs in
  `oracle_rri/app/panels/vin_utils.py`.
- VIN diagnostics now drops OBB outputs automatically when batch size > 1
  and warns the user to use batch size 1 if they need OBB predictions.

## Tests
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_vin_utils.py`

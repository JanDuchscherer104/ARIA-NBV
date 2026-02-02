# VIN diagnostics backbone keep-fields

Date: 2026-01-26

## Issue
VIN diagnostics loaded all cached backbone fields, including OBB outputs, which
caused `VinOracleBatch.collate` to fail when batching.

## Fix
- Added `DEFAULT_BACKBONE_KEEP_FIELDS` and applied it in
  `_build_experiment_config` so VIN diagnostics only decodes the fields used by
  `model_v3.py` (OBB outputs excluded).
- If a TOML provides `backbone_keep_fields`, it is preserved instead of
  overriding.

## Tests
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_vin_utils.py`

# VIN plotting refactor (v3-compatible helpers)

Date: 2026-01-26

## Scope
- Moved VIN diagnostics plotting helpers that are compatible with `model_v3.py` into
  `oracle_rri/vin/plotting.py`.
- Updated VIN diagnostics tab imports to use the new locations.
- Removed moved helper implementations from `vin/experimental/plotting.py` and trimmed its `__all__`.

## Notes
- v1/v2-only plotting helpers (tokens, alignment, SH/legacy encodings) remain in
  `vin/experimental/plotting.py`.
- `build_voxel_inbounds_figure` now flattens transformed centers to avoid shape
  mismatch with batched `PoseTW.transform`.

## Tests
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_vin_plotting_v3.py`

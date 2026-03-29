# VIN v2 geometry plotting fix

- Added a shared validity helper in `oracle_rri/oracle_rri/vin/plotting.py` to fall back to `candidate_valid` when `token_valid` is missing (VIN v2).
- Guarded `build_frustum_samples_figure` to return an empty figure when `token_valid` is absent.
- This prevents the VIN Diagnostics Geometry tab from crashing under VIN v2.

## Testing notes
- `ruff check oracle_rri/oracle_rri/vin/plotting.py` reports pre-existing lint issues (import placement, complexity, etc.).
- `pytest tests/vin/test_vin_plotting.py` under system Python 3.12 fails due to missing `power_spherical`.
- `oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_plotting.py` runs but fails because VIN backbone is disabled and no `backbone_out` is provided.

## Follow-ups
- Consider adjusting `tests/vin/test_vin_plotting.py` (or config) to enable VIN backbone or to supply cached `backbone_out` for v2/v1 debug runs.
- If VIN v2 diagnostics should plot validity fractions beyond candidate-level, consider adding token-level validity in v2 or explicitly label the colorbar as candidate-valid.

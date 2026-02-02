## Semidense projection P3D depth patch (2026-01-29)

### Summary
- Updated `_project_semidense_points` to compute metric camera-depth via
  `PerspectiveCameras.get_world_to_view_transform().transform_points(...)`
  while keeping screen-space x/y via `transform_points_screen`.
- Tightened finite/valid masks to include view-space finiteness and z>0.

### Rationale
- Aligns the documented "depth along camera axis" with actual view-space Z.
- Leverages high-level PyTorch3D transforms for clearer, less error-prone logic.

### Tests
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/vin/test_vin_model_v3_methods.py`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/integration/test_vin_v3_real_data.py`

### Notes
- `uv run pytest ...` defaulted to a Python 3.12 env missing `tomli_w`; use the
  uv-managed venv (`oracle_rri/.venv/bin/python`) for project tests.

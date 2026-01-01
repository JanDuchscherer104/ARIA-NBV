## Streamlit mesh decimation fallback (2025-11-24)

- **Issue**: Streamlit data stage crashed when `trimesh.simplify_quadric_decimation` tried to import the optional `fast_simplification` backend (ModuleNotFoundError) while decimating GT meshes via `mesh_simplify_ratio`.
- **Fix**: Added `_safe_simplify` helper in `oracle_rri/data/efm_dataset.py` that wraps quadric decimation, warns once if the backend is missing, and returns the original mesh instead of raising. Both dataset-level decimation and crop-bbox decimation now use this helper.
- **Testing**: `uv run ruff format/check oracle_rri/oracle_rri/data/efm_dataset.py`; `uv run pytest oracle_rri/tests/test_efm_dataset.py -q` (passes with local ATEK/mesh data).
- **Follow-ups**: If decimation quality/perf matters, install `fast-simplification` (pip) or add it to optional deps; otherwise the app runs with full-resolution meshes.

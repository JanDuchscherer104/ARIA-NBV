## VIN diagnostics frustum visualization (2026-01-02)

### What changed
- Added semidense projection visualization for VIN v2 in the Frustum Tokens tab (uses candidate camera projection to mark in‑view points).
- Preserved VIN v1 voxel frustum token visualization.
- Exposed a new plotting helper `build_semidense_projection_figure` for reuse.
- Updated diagnostics context to track semidense frustum availability.

### Files touched
- `oracle_rri/oracle_rri/app/panels/vin_diagnostics.py`
- `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/context.py`
- `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/tokens.py`
- `oracle_rri/oracle_rri/vin/plotting.py`

### Notes / follow-ups
- Semidense visualization requires an attached snippet (`Attach EFM snippet` in the sidebar).
- If desired, we can add a 2D image‑plane scatter view in addition to the 3D world scatter.

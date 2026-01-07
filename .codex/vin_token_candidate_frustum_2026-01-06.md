## VIN frustum tokens candidate frustum overlay (2026-01-06)

### What changed
- Added frustum overlays to the frustum token plot by reconstructing `CameraTW` + `PoseTW` from stored PyTorch3D `PerspectiveCameras`.
- Extended semidense projection plotting to optionally draw the selected candidate frustum (used in the Frustum Tokens tab when running from `vin_offline_cache`).
- Introduced a lightweight plotting stub so `_add_frusta_for_poses` can be reused without loading a full EFM snippet.

### Files touched
- `oracle_rri/oracle_rri/vin/plotting.py`
- `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/tokens.py`

### Notes / follow-ups
- Integration test `tests/vin/test_vin_plotting.py::test_vin_plotting_helpers_cpu` failed in this environment because `VinModel` requires `backbone_out` when the backbone is disabled. Re-run with a configured backbone or precomputed `backbone_out`.

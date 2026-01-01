# VIN Geometry diagnostics enhancements

- Added candidate controls in VIN Geometry: coordinate space selector (ref rig/world cam/rig), solid vs valid-fraction color mode with color picker or colorscale.
- Added candidate frusta overlay with multiselect indices, camera selection, scale, color, and axes/center toggles.
- Reference vs voxel axes now use distinct color palettes and can be toggled independently.
- Frustum rendering in VIN Geometry now uses only the final selected frame index (history trimmed in the UI).
- Backbone evidence overlay now allows choosing colorscale (hue).
- Backbone evidence points use `pts_world` when available and are filtered for finite coordinates.
- Candidate color mode now supports `loss` hue (CORAL loss per candidate when oracle RRI labels + binner are available), falling back to valid fraction with an info note if unavailable.

## Testing notes
- `ruff format oracle_rri/oracle_rri/vin/plotting.py oracle_rri/oracle_rri/app/panels.py`
- `ruff check` still fails due to pre-existing lint issues in these files.
- `pytest tests/vin/test_vin_plotting.py::test_vin_plotting_helpers_cpu` failed (missing `power_spherical` module on the current interpreter).

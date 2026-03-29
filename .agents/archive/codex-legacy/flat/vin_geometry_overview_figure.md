# VIN Geometry Overview Figure

## Change
- Added `build_geometry_overview_figure(...)` in `oracle_rri/vin/plotting.py` to combine grid bounds, reference pose axes, voxel grid axes, voxel bounds, and candidate centers colored by frustum-valid fraction into a single Plotly figure.
- Geometry tab now uses the combined figure in `oracle_rri/app/panels.py`, pulling the full `EfmSnippetView` from `VinOracleBatch`.
- Geometry tab now reuses `scene_plot_options_ui` to toggle extra layers (scene bounds, crop bounds, frustum, semidense points, GT OBBs).
 - Candidate centers now ignore `candidate_valid` masking (finite-only) so they always render; legend moved to top-left and colorbar offset to avoid overlap.

## Notes
- Uses `SnippetPlotBuilder.add_bounds_box` + `add_frame_axes_to_fig` to reuse existing plotting helpers.
- Candidate centers are transformed from rig to world using the batch reference pose.
- `scene_plot_options_ui` is wired into the geometry tab with a unique key prefix (`vin_geom`) to avoid widget collisions.
 - Candidate center arrays are reshaped to `(-1, 3)` after transform to handle batched transforms.

## Tests
- `uv run ruff format oracle_rri/vin/plotting.py oracle_rri/app/panels.py oracle_rri/vin/__init__.py ../tests/vin/test_vin_plotting.py`
- `uv run ruff check oracle_rri/vin/plotting.py oracle_rri/app/panels.py oracle_rri/vin/__init__.py ../tests/vin/test_vin_plotting.py`
- `uv run pytest ../tests/vin/test_vin_plotting.py`

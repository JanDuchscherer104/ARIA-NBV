# VIN Plotting (Plotly) Review + Changes

## Review findings (pre-change)
- `oracle_rri/oracle_rri/vin/plotting.py` uses matplotlib, saves PNGs, and returns only paths. This limits interactivity and makes Streamlit integration awkward (no in-memory figures).
- `_plot_sh_components` uses `zip(..., strict=True)` with a fixed 2×2 subplot layout; when `lmax < 1`, the list of components is shorter and `zip(..., strict=True)` can raise, producing a hard failure.
- `_plot_shell_descriptor_concept` assumes at least one candidate (`u[0]`, `f[0]`, `r[0]`) without guarding against empty inputs.
- Plot outputs are always written to disk; there is no lightweight “build figures” path for UI reuse.

## Changes applied
- Rewrote VIN plotting to use Plotly end-to-end and added figure-building helpers for VIN geometry diagnostics.
- Added Plotly visuals for voxel frame bounds, candidate directions, token validity, frustum samples, alignment histograms, field slices, field/token histograms, backbone evidence, and voxel roundtrip residuals.
- Integrated new visuals into the Streamlit VIN diagnostics panel, with dedicated tabs for geometry, fields, frustum tokens, backbone evidence, transforms, and SH/Fourier features.
- Added an integration test with real data to ensure all new Plotly builders return figures.
- Fixed `build_voxel_frame_figure` to handle batched corner transforms from `PoseTW.transform`.

## Potential issues / risks
- Output format changed from PNG to HTML for saved artifacts (CLI/experiment logs). Consumers expecting PNGs will need to adjust.
- Plotly HTML files can be larger; consider adding an optional static image export (requires kaleido) if needed.
- Some 3D scatters can be heavy for large voxel grids; current logic downsamples but may still need user controls for max points.

## Tests
- `uv run ruff format oracle_rri/vin/plotting.py oracle_rri/app/panels.py ../tests/vin/test_vin_plotting.py`
- `uv run ruff check oracle_rri/vin/plotting.py oracle_rri/app/panels.py ../tests/vin/test_vin_plotting.py`
- `uv run pytest ../tests/vin/test_vin_plotting.py ../tests/vin/test_vin_diagnostics.py`
  - Result: 5 passed (warnings from lightning_fabric/pkg_resources deprecations).

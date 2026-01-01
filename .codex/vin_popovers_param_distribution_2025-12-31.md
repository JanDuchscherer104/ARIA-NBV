# VIN diagnostics popovers + parameter distribution

## Summary
- Expanded VIN diagnostics popovers with VIN v2-specific explanations for scatter, feature dims, scene field channels, and feature norms.
- Added new popovers for pose descriptor and scene field slices.
- Added a trainable-parameter distribution bar chart grouped by top-level VIN submodule.

## Notes / Findings
- Parameter grouping is based on the first component of `named_parameters()` (e.g., `pose_encoder_lff`, `global_pooler`, `head_mlp`, `head_coral`, `pose_log_scale`).
- Frozen parameters (e.g., EVL backbone) are excluded so the plot reflects trainable capacity.

## Potential Issues
- Top-level grouping may hide large sub-structure differences inside modules (e.g., `backbone` internals).
- If a model exposes trainable params under unusual names (no dot), they appear as their full name (single bar).

## Suggestions
- Add a toggle to switch between trainable-only vs total parameter counts.
- Consider a secondary grouping level (e.g., `module.submodule`) for deeper breakdowns when diagnosing capacity bottlenecks.

## 2025-12-31 Update: FF encodings + pos-grid error
- Added detailed FF Encodings popovers in the VIN diagnostics tab (pose vector, LFF empirical plots, pose encoding PCA, position grid slices, position grid PCA).
- Fixed positional grid PCA error by replacing `np.linalg.vector_norm` with `np.linalg.norm` in `oracle_rri/oracle_rri/vin/plotting.py`.

### Testing
- `ruff format oracle_rri/oracle_rri/app/panels.py oracle_rri/oracle_rri/vin/plotting.py`
- `ruff check oracle_rri/oracle_rri/app/panels.py oracle_rri/oracle_rri/vin/plotting.py` failed due to pre-existing lint violations in these files.
- `pytest oracle_rri/oracle_rri/app/panels.py oracle_rri/oracle_rri/vin/plotting.py` timed out during collection.

## 2025-12-31 Update: Additional panel popovers
- Added academic info popovers across data, candidates, depth, RRI, VIN diagnostics (geometry/tokens/evidence/transforms), and offline stats sections in `oracle_rri/oracle_rri/app/panels.py`.

## 2025-12-31 Update: Additional diagnostics popovers
- Added concise academic popovers for data overlays, candidate plots, depth renders, RRI metrics, geometry/tokens/evidence/transforms, and offline stats.

### Testing
- `ruff format oracle_rri/oracle_rri/app/panels.py`
- `ruff check oracle_rri/oracle_rri/app/panels.py` (fails due to pre-existing lint violations)
- `pytest oracle_rri/oracle_rri/app/panels.py` failed during collection: missing `power_spherical`.

## 2025-12-31 Update: Transforms diagnostics
- Added SE(3) closure, voxel in-bounds ratio, and pos-grid linearity plots in the Transforms tab.
- Implemented plotting helpers: `build_se3_closure_figure`, `build_voxel_inbounds_figure`, `build_pos_grid_linearity_figure` in `oracle_rri/oracle_rri/vin/plotting.py` and wired them into `oracle_rri/oracle_rri/app/panels.py`.

### Testing
- `ruff format oracle_rri/oracle_rri/app/panels.py oracle_rri/oracle_rri/vin/plotting.py`
- `ruff check oracle_rri/oracle_rri/app/panels.py oracle_rri/oracle_rri/vin/plotting.py` failed due to existing lint violations.
- `pytest oracle_rri/oracle_rri/app/panels.py` failed during collection: missing `power_spherical`.

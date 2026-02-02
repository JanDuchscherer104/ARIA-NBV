# Semi-dense Feature Branch Slides + Diagnostics Notes (2026-01-29)

## Goal
Update `docs/typst/slides/slides_4.typ` so the VINv3 *semi-dense feature branches* are accurate and self-contained, and document what the Streamlit “Tokens” tab shows for semi-dense projections.

## Key code sources
- `oracle_rri/oracle_rri/vin/model_v3.py`
  - `SEMIDENSE_PROJ_FEATURES = (coverage, empty_frac, semidense_candidate_vis_frac, depth_mean, depth_std)`
  - `_encode_semidense_projection_features(...)`: reliability-weighted visibility + depth moments
  - `_encode_semidense_grid_features(...)`: unweighted (valid-only) occupancy/depth mean/std grid + tiny CNN
- `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/tokens.py`
  - “Semidense projection feature maps (v3)” and “Semidense CNN grid inputs”
- `oracle_rri/oracle_rri/vin/plotting.py`
  - `build_semidense_projection_feature_maps`: per-cell `counts`, `weights`, `depth_mean`, `depth_std`
  - `build_semidense_cnn_grid_maps`: per-cell `occupancy`, `depth_mean`, `depth_std` (matches CNN inputs)

## Slide updates made
File: `docs/typst/slides/slides_4.typ`

- Expanded **“Semidense projections: visibility + grid CNN”** with:
  - explicit reliability-weight formula (track-length + inverse-depth-uncertainty),
  - the exported scalar tuple `semidense_proj=(...)`,
  - a note that diagnostics renders both (a) CNN input grids and (b) reliability-weighted feature maps.
- Expanded **Branch 6** to clarify:
  - subsampling budget (`semidense_proj_max_points`),
  - per-feature semantics (grid occupancy coverage, weighted visibility, weighted depth moments),
  - which config scalars normalize the reliability weights.
- Expanded **Branch 7** to clarify:
  - config toggles/dims (`semidense_cnn_enabled`, `semidense_cnn_channels`, `semidense_cnn_out_dim`),
  - that diagnostics renders the *exact* grids fed into the CNN.

## Notes / follow-ups
- The CNN grid path is currently **unweighted** (valid-only). If we want it to reflect `n_obs` and `1/σ_d`, the grid accumulation would need reliability weights similar to the scalar path.
- The diagnostic “feature maps” (counts/weights/depth_mean/depth_std) are **interpretability aids**; the head consumes the scalar `semidense_proj`, and (optionally) the 3-channel CNN grid.


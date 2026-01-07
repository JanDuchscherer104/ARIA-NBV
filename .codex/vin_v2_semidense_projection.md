# VIN v2 semidense projection features

## Summary
- Added per-candidate semidense projection features (coverage, empty_frac, valid_frac, depth_mean, depth_std) computed via PyTorch3D `transform_points_screen`.
- Injected these features by FiLM-modulating `global_feat` and concatenating to the head features.
- Added config knobs for `semidense_proj_grid_size` and `semidense_proj_max_points`.
- Exposed `semidense_proj` in `VinV2ForwardDiagnostics` and VIN v2 summary.
- Documented the injection strategy in `docs/contents/impl/vin_v2_feature_proposals.qmd`.

## Notes / Suggestions
- Projection features rely on `p3d_cameras.image_size`; when missing, the features currently default to zeros. Consider asserting or logging when this happens in training to avoid silent degradation.
- The projection uses a coarse grid; for higher fidelity, consider adding optional multi-scale grids or angular bins as a follow-up ablation.
- We weight depth stats by `inv_dist_std` when available; if we want stronger uncertainty awareness, consider adding a separate uncertainty scalar (mean/var of inv_dist_std).

## Tests
- `oracle_rri/tests/vin/test_vin_model_v2_gradients.py::test_semidense_projection_features_shape`
- `oracle_rri/tests/integration/test_vin_v2_real_data.py::test_vin_v2_forward_real_snippet_produces_scores`

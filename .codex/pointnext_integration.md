# PointNeXt-S optional semidense encoder

## Summary
- Added optional PointNeXt-S adapter in `oracle_rri/oracle_rri/vin/model_v2.py` with YAML-based OpenPoints config loading and checkpoint support.
- Subsamples semidense points to `max_points` (default 3000) via `EfmPointsView.collapse_points` before encoding.
- Added optional config hook (`VinModelV2Config.point_encoder`) and integrated semidense embedding into the VIN v2 head inputs.
- Updated `docs/contents/impl/vin_v2_feature_proposals.qmd` and `docs/references.bib` with PointNeXt/OpenPoints references.

## Notes / Suggestions
- If using the PointNeXt-S encoder, ensure OpenPoints is installed and provide the model zoo YAML + checkpoint path; otherwise the adapter raises a clear error.
- Consider adding a config flag for candidate-conditioned point encoding (camera-frame points) if global embeddings prove too coarse.
- Running the integration tests requires the real ASE snippet fixture; ensure the test assets are available before relying on CI.

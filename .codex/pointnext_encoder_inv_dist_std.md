# PointNeXt semidense feature channel (inv_dist_std)

## Summary
- `EfmPointsView.collapse_points` now returns `Tensor["K 4"]` when `include_inv_dist_std=True` (XYZ + inv_dist_std), preserving aligned features without time-collapsing uniqueness.
- `VinModelV2._encode_semidense_features` now requests `include_inv_dist_std=True`, applies pose transforms to XYZ only, and concatenates the scalar feature back before the point encoder.
- `PointNeXtSEncoder` now supports inputs with per-point features (B,N,C), splits XYZ vs features, pads/truncates to expected input channels (from config), and passes features via `forward_cls_feat`.
- Ensured XYZ is contiguous and projection layer is moved to the correct device.

## Files touched
- `oracle_rri/oracle_rri/data/efm_views.py`
- `oracle_rri/oracle_rri/vin/model_v2.py`
- `oracle_rri/oracle_rri/vin/pointnext_encoder.py`

## Notes
- For `include_inv_dist_std=True`, duplicates are not removed (to preserve feature alignment). If de-duplication is desired later, a feature-aware reduction strategy is needed.

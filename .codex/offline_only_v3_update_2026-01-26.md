# offline_only.toml v3 compatibility update (2026-01-26)

## Summary
- Updated `.configs/offline_only.toml` to be compatible with `oracle_rri/oracle_rri/vin/model_v3.py`.
- Removed v2-only fields (PointNeXt, trajectory encoder, semidense frustum settings, voxel-valid feature concat).
- Kept overlapping hyperparameters (field_dim/head depth/dropout, pool grid, semidense projection grid/points).

## Notes
- v3 still uses semidense projection features (no PointNeXt), so `semidense_include_obs_count=true` in the cache config remains appropriate.
- Remaining correctness risk: CW90 camera correction mismatch for `p3d_cameras` when `apply_cw90_correction=true`.

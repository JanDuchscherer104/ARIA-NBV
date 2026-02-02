# Offline-only config aligned to best VIN v2 trial (2026-01-26)

## Scope
- Updated `.configs/offline_only.toml` to match W&B best trial `wsfpssd8` settings.

## Key updates
- Data: batch_size=8, num_workers=15.
- Optimizer: weight_decay=0.013244782071249196 (lr stays 3e-4).
- OneCycleLR: max_lr=0.0001830227950357318, pct_start=0.1230097043157138, div_factor=44.46383233720467, final_div_factor=316.4875424977932, base_momentum=0.85, max_momentum=0.95.
- VIN v2 knobs: field_dim=24, field_gn_groups=7, head_num_layers=1, head_dropout=0.0071317082333434145, global_pool_grid_size=5.
- Enable v2 features: use_point_encoder/use_traj_encoder/enable_semidense_frustum, semidense_obs_count_norm="none", semidense_visibility_embed=true, semidense_frustum_mask_invalid=true, semidense_frustum_max_points=512, semidense_proj_grid_size=12, semidense_proj_max_points=4096, use_voxel_valid_frac_feature=true, use_voxel_valid_frac_gate=false.
- PointNeXt: out_dim=192, max_points=5000 (cfg + checkpoint paths unchanged from prior template).
- Coverage anneal epochs adjusted to 3 (match run config).

## Follow-ups / remaining inconsistencies
- Current Lightning config defaults to `VinModelV3Config`; v2-only fields in the TOML will be ignored unless you switch to the experimental v2 config class explicitly.
- CW90 camera correction mismatch persists in v2/v3: `apply_cw90_correction=True` rotates poses but not `p3d_cameras` used for projection features.

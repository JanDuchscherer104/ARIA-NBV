# VIN v2 refactor: module split

## Summary
- Split VIN v2 helpers into `oracle_rri/oracle_rri/vin/vin_v2_utils.py` (dataclasses + utility functions) and `oracle_rri/oracle_rri/vin/vin_v2_modules.py` (PoseConditionedGlobalPool).
- Updated `oracle_rri/oracle_rri/vin/model_v2.py` to import and use the new modules; logic unchanged.

## Files touched
- `oracle_rri/oracle_rri/vin/model_v2.py`
- `oracle_rri/oracle_rri/vin/vin_v2_utils.py`
- `oracle_rri/oracle_rri/vin/vin_v2_modules.py`

## Potential issues / follow-ups
- CW90 correction still only applies to `pose_world_cam` and `pose_world_rig_ref`; `t_world_voxel` and `pts_world` remain in the original world frame, which can cause frame inconsistency if `apply_cw90_correction=True`.
- Consider removing the unused `pool` attribute in `PoseConditionedGlobalPool` (it currently uses `functional.adaptive_avg_pool3d` instead).
- If cached batches are used with `point_encoder`, ensure `EfmSnippetView.from_cache_efm` handles empty `efm` dicts gracefully or add a fallback.

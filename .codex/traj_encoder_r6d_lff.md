# Trajectory encoder (R6D + LFF)

## Summary
- Added `oracle_rri/oracle_rri/vin/traj_encoder.py` implementing `TrajectoryEncoder` that encodes `EfmTrajectoryView.t_world_rig` using `R6dLffPoseEncoder`.
- Added `TrajectoryEncoderConfig` (config-as-factory) and `TrajectoryEncodingOutput` with optional pooling (`mean`, `final`, `none`).
- Exported new classes from `oracle_rri/oracle_rri/vin/__init__.py`.
- Documented the new trajectory encoder in `docs/contents/impl/vin_nbv.qmd`.

## Files touched
- `oracle_rri/oracle_rri/vin/traj_encoder.py`
- `oracle_rri/oracle_rri/vin/__init__.py`
- `docs/contents/impl/vin_nbv.qmd`

## Potential follow-ups
- Decide whether trajectory encoding should be relative to a reference pose rather than absolute `t_world_rig`.
- Add unit tests for batch handling and pooling modes.

## Config compatibility fix
- Restored `VinModelV2Config.tf_pos_grid_in_candidate_frame` as a deprecated field to allow older TOML configs to load.

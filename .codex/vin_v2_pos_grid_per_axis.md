# VIN v2 pos_grid normalization

## Summary
- Switched `pos_grid_from_pts_world` normalization to per-axis scaling (0.5 * span per axis) instead of max-span isotropic scaling.
- Updated docstring to reflect per-axis normalization.

## Files touched
- `oracle_rri/oracle_rri/vin/vin_v2_utils.py`

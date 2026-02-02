# VIN diagnostics: semidense projection feature maps

## Summary
- Added semidense projection feature map visualization (counts/weights/depth mean/std) for VINv3.
- New plotting helpers compute projection grids using candidate camera and semidense points.
- VIN diagnostics now shows the maps under the Frustum Tokens tab when v3 debug outputs are present.

## Tests
- `oracle_rri/tests/vin/test_vin_plotting_v3.py`

## Notes
- Maps use cached VIN snippet points when available and fall back to EFM semidense collapse with inv_dist_std + obs_count.
- Camera index is flattened when batch size > 1.

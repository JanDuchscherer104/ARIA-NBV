# VIN evidence plots: use cached voxel centers

## Summary
- Evidence plotting now uses `backbone_out.pts_world` when available to map voxel indices to world points.
- Falls back to pose/extent transform if cached centers are unavailable or shape mismatched.

## Motivation
- Some cache samples showed empty evidence plots despite populated voxel tensors; cached voxel centers are more robust than recomputing from `T_world_voxel`.

## Tests
- `oracle_rri/tests/vin/test_vin_plotting_v3.py`

## Notes
- Both `build_scene_field_evidence_figures` and `build_backbone_evidence_figures` share the new mapping helper.

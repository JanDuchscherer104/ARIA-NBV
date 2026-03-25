# Voxel Extent Horizontal Expansion Notes

## Findings
- EVL's voxel grid is defined by `video_backbone3d.voxel_extent` in the EVL config; this drives the lifter's voxel grid creation and is surfaced as `voxel_extent` in EVL outputs (`external/efm3d/efm3d/model/lifter.py`, `external/efm3d/efm3d/model/evl.py`).
- The lifter enforces cubic voxels: `(x_max-x_min)/vW == (y_max-y_min)/vH == (z_max-z_min)/vD` (assertion in `external/efm3d/efm3d/model/lifter.py`).
- Changing `voxel_extent` without compensating `voxel_size` changes `voxel_meters`, which EVL uses for OBB decoding (`offset_max`, `splat_sigma`) and affects the physical scale of features.
- VIN consumes `voxel_extent` to normalize positions and sample the voxel field; no code changes are needed if EVL outputs a consistent extent, but candidate validity fractions and sampling coverage will change.

## Risks / Potential Problems
- If x/y extent grows while z extent and `voxel_size` stay fixed, the lifter's cubic-voxel assertion fails.
- If you change extent but keep voxel_size fixed (or change all extents equally), the physical voxel size changes, which can degrade EVL inference because weights are trained at the original scale.
- Increasing `voxel_size` to keep voxel_meters constant increases memory/compute and may require batch-size or feature-dim adjustments.

## Suggestions
- If you want larger horizontal coverage **without changing voxel_meters**, increase `voxel_size` in H/W proportionally: e.g., for 6m x 6m with 0.0833m voxels, use `voxel_size: [48, 72, 72]` while keeping z extent at 4m.
- If you want to keep `voxel_size` fixed, expand **all** extents equally (x/y/z) to preserve cubic voxels, but expect a change in effective resolution.
- If you plan to use EVL OBB outputs, prefer keeping `voxel_meters` consistent to avoid scale drift in decoded boxes.
- Update any other EVL configs you rely on (e.g., `external/efm3d/efm3d/config/evl_inf_desktop.yaml` for EFM3D scripts) so inference and debugging stay aligned.

## References (local)
- `external/efm3d/efm3d/model/lifter.py`
- `external/efm3d/efm3d/model/evl.py`
- `.configs/evl_inf_desktop.yaml`
- `oracle_rri/oracle_rri/vin/model.py`
- `oracle_rri/oracle_rri/vin/model_v2.py`

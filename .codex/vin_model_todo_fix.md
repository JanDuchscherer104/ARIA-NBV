# Resolve VIN `model.py` TODO/FIXME (2025-12-19)

## Goal

Remove the remaining inline `TODO`/`FIXME` markers in `oracle_rri/oracle_rri/vin/model.py` by:

- eliminating “silent fallback” behavior (implicit reference-pose guessing, optional camera inputs),
- simplifying the VIN call contract, and
- keeping training/inference aligned with the PyTorch3D depth-rendering camera model.

## Changes

### VIN API tightening

- `VinModel.forward(...)` now requires:
  - `candidate_poses_world_cam: PoseTW` (world←cam; `(N,12)` or `(B,N,12)`),
  - `reference_pose_world_rig: PoseTW` (world←rig; `(12,)` or `(B,12)`),
  - `p3d_cameras: PerspectiveCameras` (same ordering as candidates).
- Removed the “training special-case” input `candidate_poses_camera_rig` and its branching logic.
- Removed the internal reference-pose lookup from the EFM dict (`_get_reference_pose_world_rig`) to prevent silent mismatches.

### Frustum query simplification

- Removed the fixed-FOV fallback frustum grid and its config field (`frustum_fov_deg`).
- Frustum sampling is now exclusively driven by PyTorch3D unprojection (`PerspectiveCameras.unproject_points(..., from_ndc=True, world_coordinates=True)`), matching the depth renderer’s convention.

### Minor cleanup

- Inlined masked mean pooling inside `VinModel._pool_candidates` (removed `_safe_mean_pool` free function).
- Kept a “re-verify if conventions change” note for voxel/world transforms, but removed the `FIXME:` tag.

### Pipeline + tests updates

- Updated `VinOracleBatch` to drop `candidate_poses_camera_rig` and updated `VinLightningModule` call site accordingly.
- Updated real-data integration tests to explicitly build `PerspectiveCameras` for candidates.
- Updated lightweight helper tests to validate `_build_frustum_points_world_p3d` shape (since `_build_frustum_points_cam` was removed).

## Acceptance / verification

- `rg "TODO|FIXME" oracle_rri/oracle_rri/vin/model.py` returns no matches.
- `ruff format` + `ruff check` pass for touched VIN files.
- `pytest` passes including real-data:
  - `tests/vin/test_vin_model_integration.py`
  - `oracle_rri/tests/integration/test_vin_real_data.py`
  - `oracle_rri/tests/integration/test_vin_lightning_real_data.py::test_vin_lightning_fit_runs_real_data_smoke`

## Notes / follow-ups

- This change intentionally removes convenience defaults to reduce “silent wrong” failure modes; callers must now always provide the reference pose and the corresponding PyTorch3D cameras.
- If you want to support non-rendered inference later, add a small utility to construct `PerspectiveCameras` from `(PoseTW world←cam, CameraTW intrinsics)` in a shared rendering helper (kept out-of-scope here).


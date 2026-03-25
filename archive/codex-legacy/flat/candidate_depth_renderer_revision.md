# Candidate Depth Renderer – Issues & Fix Plan

Context: CandidateSamplingResult now exposes per-candidate `CameraTW` (intrinsics + `T_camera_rig` in reference frame) and `reference_pose` (reference2world). The current renderer still follows the pre-refactor contract.

Key issues observed in `oracle_rri/rendering/candidate_depth_renderer.py`:
- Uses snippet camera (`sample.get_camera`) instead of `candidates.views` for intrinsics; ignores the generator-selected stream/frame.
- Treats `candidates.views` as poses; no composition with `reference_pose`, so PyTorch3D receives the wrong extrinsics.
- Still filters by `mask_valid` even though `views` is post-pruning; indices can diverge from `views`.
- Keeps frame-based calib selection/downscaling of an `EfmCameraView`; unnecessary and can desync intrinsics if `resolution_scale` is applied.
- Returns `mask_valid` from input rather than the rendered subset; misleading for downstream plotting/metrics.

Secondary gaps in `oracle_rri/rendering/pytorch3d_depth_renderer.py`:
- `_camera_intrinsics` selects only the first entry of a batched `CameraTW`; per-candidate intrinsics would be dropped.
- Depth buffer is camera-Z; callers must treat it accordingly when back-projecting.

Proposed fixes:
- Build world poses via `T_world_cam = candidates.reference_pose @ PoseTW(candidates.views.T_camera_rig)`, then pass to the renderer.
- Use `candidates.views` for intrinsics; if subsampling, slice both poses and cameras together. Reserve `mask_valid` only for optional logging.
- If resolution scaling is desired, call `CameraTW.scale_to_size` on the sliced `views`; drop `_downscale_camera_view`.
- In the PyTorch3D renderer, allow batched intrinsics by expanding `f`, `c`, `image_size` from the provided `CameraTW` instead of taking only the first row.
- Add a debug sanity check for min camera-space z after the world→view inversion to catch pose-frame mixups early.

Depth → world points with `CameraTW.unproject`:
1) Build `T_world_cam` as above.
2) Generate a pixel grid, call `rays_cam, valid = cam.unproject(grid)`.
3) Multiply rays by depth (camera-Z) to get cam-frame points; mask where depth == zfar or invalid.
4) Transform with `T_world_cam.transform(pts_cam)` to align with semi-dense clouds.

This aligns the renderer with the new CandidateSamplingResult contract and prepares clean unprojection for downstream NBV debugging.

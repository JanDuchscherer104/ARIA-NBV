# VIN pose frame consistency (offline cache)

Date: 2026-01-26

## Findings
- Offline cache sample check: `candidate_poses_world_cam` matches `p3d_cameras` (max abs R diff 0, max abs T diff ~1e-6).
- Undoing `rotate_yaw_cw90` on poses only (VinModelV3 `apply_cw90_correction=True`) breaks alignment (max abs R diff ~1.25, max abs T diff ~12), so semidense projection features are in a different camera frame than pose/global context.

## Implications
- For cached datasets (no recompute), keep `apply_cw90_correction=False` to preserve internal consistency.
- If we insist on undoing CW90, rotate `p3d_cameras` (and `reference_pose_world_rig`) in lockstep at batch-prep time, and add a consistency check.

## Follow-ups
- Consider a VinOracleBatch frame-normalization helper that applies CW90 to poses + `p3d_cameras` together.
- Add a one-time warning/assertion in VinModelV3 for `p3d_cameras` vs `candidate_poses_world_cam` consistency.

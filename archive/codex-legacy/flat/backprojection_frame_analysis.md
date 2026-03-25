# Depth backprojection frame analysis (2025-12-01)

Observations from the Streamlit candidate viewer:
- Frusta and RGB thumbnails look geometrically consistent, but the depth hit back-projection is far off (diagonal spray outside the room).
- Some room shells (walls/floor/ceiling) remain unrendered even with backface culling disabled.

What the current pipeline does
- CandidateDepthRenderer builds poses as `T_world_cam = T_world_ref @ T_cam_ref.inverse()` where `T_cam_ref` is `views.T_camera_rig` (camera ← rig) and `reference_pose` is `world ← rig`. (oracle_rri/rendering/candidate_depth_renderer.py)
- PyTorch3DDepthRenderer inverts that pose (`poses_cw = poses.inverse()`) and feeds R/T to `PerspectiveCameras`; depth is taken as `pts_cam[:, 2]` after world→camera. (oracle_rri/rendering/pytorch3d_depth_renderer.py)
- `add_depth_hits` back-projects with a manual pinhole model: `x = (u - cx)/fx * z`, `y = (v - cy)/fy * z`, then `pose.transform(...)` with `pose` assumed to be `world ← cam`. Camera extrinsics are ignored and, when a batch camera is passed, only `camera[0]` is used. (oracle_rri/rendering/plotting.py::_backproject_depth)
- Frusta drawing uses `camera.unproject` and whatever pose the caller provides. In the data plots we previously built poses as `rotate_yaw_cw90(T_world_rig @ T_cam_rig.inverse())`, i.e. a +90° yaw tweak. (oracle_rri/data/plotting.py)

Likely sources of the misalignment
- Pixel-axis convention: the manual back-projection assumes x→right, y→down, but our camera frames are documented as LUF (x left, y up, z forward). That should introduce sign flips (x, y) when mapping pixels to camera rays; `camera.unproject` already bakes the correct convention, so the manual pinhole is probably using the wrong handedness.
- Pose/camera pairing: `_backproject_depth` drops per-candidate intrinsics/extrinsics by forcing `cam_single = camera[0]`; this mismatches the per-candidate pose batch.
- The optional +90° yaw adjustment used in frustum plots is not applied in depth rendering/back-projection, so the pose supplied to `_backproject_depth` may be in a slightly different frame than the one used for visual frusta in other views.

Next checks to confirm
- For one candidate, compare rays from `_backproject_depth` against `camera.unproject` + `pose.transform` (with/without yaw flip); log a few pixel corners/center to see which frame matches the rendered mesh.
- Verify PyTorch3D camera handedness against PoseTW by projecting unit axes; if signs differ, incorporate the flips in the back-projection or switch to `camera.unproject`.
- Use the cached per-candidate cameras instead of `camera[0]` in `_backproject_depth` to eliminate batch mismatch as a factor.
- Inspect hit/miss masks to see if missing walls relate to zfar or backface handling; try `faces_per_pixel>1` in the renderer as a quick sanity check once the frame issue is resolved.

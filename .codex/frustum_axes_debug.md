# Frustum & Axes Debug Notes (2025-11-23)

Goal: fix Plotly frustum/axes orientation regressions after LUF cleanup, with a record of all changes made.

Reference implementations:
- efm3d frustum: `external/efm3d/efm3d/utils/viz.py::render_frustum`
  - corners from `valid_radius * 0.7071`
  - unproject with `cam.unproject`
  - pose: `T_wc = T_wr @ cam.T_camera_rig.inverse()` (RDF frame)
  - no display flips/roll/yaw
- CameraTW tools: `rotate_90(clock_wise/ccw)` rotates camera extrinsics by 90° about +Z.
- Pose math reference: `external/efm3d/efm3d/aria/pose.py` (PoseTW transforms)
- Camera model reference: `external/efm3d/efm3d/aria/camera.py` (CameraTW, valid_radius, rotate_90)
- ATEK viz (for conceptual parity): `external/ATEK/atek/viz/atek_visualizer.py` uses similar pose logic (matrix3x4) without display flips.

Plotly path experiments (chronological):
1) Switched plotting docstrings to LUF; added display-only LUF→display rotation in `pose_for_display`.
2) Made `view_axes_from_points` emit LUF (left/up/forward) while staying right-handed.
3) Reverted frustum hat to efm3d style (edge-based) and radius *0.7071.
4) Removed `_orientation_fix` and centralized `world_from_rig_camera_pose`; added aliases `ASEDatasetConfig` and `gt_mesh`.
5) Tried RDF flip (diag -1,-1,1) for plotting: hat/axes still off.
6) Tried raw pose (no display) for frusta: axes OK, hat sideways.
7) Tried `cam.rotate_90(clock_wise=True)` before unproject: worse.
8) Restored display transform (gravity align + roll -90°, yaw +90°) for frusta/axes; kept efm3d corner/hat; removed RDF flip and rotate_90. Current state.

Key toggles we tried:
- Display flip matrix: X-only flip (diag(-1,1,1)); full RDF flip (diag(-1,-1,1)); none.
- Hat source: camera +Y vs image-plane top edge. Best match is efm3d style (edge).
- Pose path: raw LUF vs display-adjusted pose_for_display vs LUF→RDF.
- Camera rotation: rotate_90 CW — reverted.

Tests:
- `pytest oracle_rri/tests/pose_generation/test_pose_generation.py` passes throughout.
- `tests/integration/test_frames_real_data.py` axis sanity passes; real-data roundtrips skip if data missing.

Open hypothesis:
- Plotly’s default axis orientation plus our display roll/yaw may still induce a +/−90° yaw about camera Y. If visuals remain twisted, flip the display yaw sign in `pose_for_display` (±90° about Y) as a final tweak.

Where to adjust next:
- `oracle_rri/utils/frames.py::pose_for_display` (yaw sign).
- Keep frustum building aligned to efm3d: no extra camera rotate_90, no RDF flip; hat from edge; radius *0.7071.

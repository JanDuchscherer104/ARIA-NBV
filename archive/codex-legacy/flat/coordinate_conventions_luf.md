# Coordinate Convention Corrections (2025-11-23)

## Findings
- Project Aria/EFM3D cameras are **LUF (left–up–forward)**. Earlier plotting assumed RDF, causing mirrored frusta/axes.
- `pose_for_display` applies a display-only LUF→display 180° rotation about +Z before viewer roll/yaw; rotations stay in SO(3).
- Plotting/docstrings/UI copy now call out LUF; candidate-generation docs reference the correct convention. Frame helper outputs a left/up/forward basis (right-handed).

## Changes Made
- Added `LUF_TO_DISPLAY_ROT` in `pose_for_display`; updated plotting docstring accordingly.
- Reworded candidate-generation/doc UI copy from RDF→LUF and added explanatory notes.
- Updated `view_axes_from_points` to emit a LUF-labelled basis (left, up, forward) while keeping rotations right-handed.
- Added back-compat aliases `ASEDatasetConfig` and `gt_mesh` to satisfy tests/UI.
- Fixed frustum corner scaling (inscribed square radius / sqrt(2)) in plotting.

## Gaps / Next Steps
- Revisit depth/candidate renderers and orientation helpers (e.g., `_orientation_fix`) to remove any residual ad-hoc axis flips.
- Run broader test sweep (pose_generation suite now passes; warning remains about non-writable NumPy array when reading mesh bounds).
- New integration tests added (`tests/integration/test_frames_real_data.py`) to sanity-check LUF axes and roundtrips; real-data tests auto-skip if ASE shards absent.

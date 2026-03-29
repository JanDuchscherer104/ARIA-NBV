# Update `summarize_vin.py` for current VIN (2025-12-19)

## Goal

Keep `oracle_rri/scripts/summarize_vin.py` in sync with the current VIN implementation:

- VIN now requires explicit `reference_pose_world_rig` + `p3d_cameras` for frustum queries.
- VIN feature breakdown should include **pose/global/local** features and candidate validity.

## Changes

- Added utilities to build `pytorch3d.renderer.cameras.PerspectiveCameras` directly from:
  - candidate `PoseTW` (world←cam), and
  - `CameraTW` intrinsics (pinhole; uses a single-frame slice for this script).
- Expanded the “VIN feature concat” section to compute:
  - `pose_enc` (ShellShPoseEncoder),
  - `global_feat` (mean pooled voxel field, if enabled),
  - `local_feat` (masked-mean voxel sampling along frustum points unprojected via PyTorch3D cameras),
  - `candidate_valid` from frustum validity fraction.
- Now prints VIN output shapes (logits/prob/expected/expected_normalized) computed from head logits.
- Added `torchsummary` for `vin.field_proj` (use pre-projection field as input).

## Quick usage

Run (CPU example):

`oracle_rri/.venv/bin/python oracle_rri/scripts/summarize_vin.py --device cpu --scene-id 81286 --num-candidates 3`


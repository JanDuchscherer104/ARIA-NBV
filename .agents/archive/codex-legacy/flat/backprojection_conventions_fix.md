# Fix: Candidate Frusta vs Backprojected Point-Cloud Mismatch

## Problem (from `docs/contents/prompt_render_issues.md`)

Candidate-view depth maps were rendered, but the backprojected point clouds:

- often appeared **in front of the reference pose regardless of candidate look direction**, and
- did **not align with the plotted candidate frusta**, producing “look-through-walls” artefacts and many “no hit”
  pixels for poses that should be valid.

## Root Cause

Two independent convention bugs compounded:

1. **Extrinsics convention mismatch (PoseTW ↔ PyTorch3D)**
   - `efm3d.aria.PoseTW` stores rotations in the standard “column-vector” convention (`PoseTW.transform()` applies
     `p @ R^T + t`).
   - PyTorch3D’s `Transform3d` / `PerspectiveCameras` world→view mapping uses **row vectors**
     (`X_cam = X_world R + T`).
   - Passing `PoseTW.R` directly as `PerspectiveCameras.R` therefore applies the wrong rotation (missing transpose),
     which makes camera frusta and unprojected points disagree.

2. **Incorrect pixel-coordinate flips during unprojection**
   - `PerspectiveCameras.unproject_points(..., from_ndc=False)` expects screen-space pixel coordinates in the usual
     image convention (origin top-left, +x right, +y down) for `in_ndc=False` cameras.
   - Flipping pixels as `x := (w-1)-x`, `y := (h-1)-y` mirrors the unprojection and causes points to appear on the
     wrong side of the frustum.

## Changes Made

- `oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py`
  - Pass `R = poses.inverse().R.transpose(-1, -2)` into `PerspectiveCameras` so PyTorch3D’s row-vector mapping matches
    `PoseTW.transform()`.
  - Removed the earlier “flip X/Y axes” hack (it masked the true issue and broke consistency).

- `oracle_rri/oracle_rri/rendering/unproject.py`
  - Removed `w/h` pixel flips; unprojection now uses raw `(x_px, y_px)` coordinates.

- `oracle_rri/oracle_rri/rendering/candidate_pointclouds.py`
  - Removed `w/h` pixel flips in the batched backprojection path.

## Tests Added

- `oracle_rri/tests/rendering/test_depth_backprojection_conventions.py`
  - Unit regression tests for:
    - correct world→view extrinsics mapping (`PoseTW.inverse().transform` ≈ PyTorch3D `world_to_view`), and
    - correct pinhole backprojection signs in pixel space (+x right, +y down).

- `oracle_rri/tests/integration/test_render_backproject_real_data.py`
  - Real-data integration test (ASE sample with mesh):
    - render a depth map,
    - backproject pixels left/right of `cx`,
    - assert camera-frame X sign matches the pixel side.

## Commands Run (local)

From repo root:

- `oracle_rri/.venv/bin/ruff format oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py oracle_rri/oracle_rri/rendering/unproject.py oracle_rri/oracle_rri/rendering/candidate_pointclouds.py oracle_rri/tests/rendering/test_depth_backprojection_conventions.py oracle_rri/tests/integration/test_render_backproject_real_data.py`
- `oracle_rri/.venv/bin/ruff check oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py oracle_rri/oracle_rri/rendering/unproject.py oracle_rri/oracle_rri/rendering/candidate_pointclouds.py oracle_rri/tests/rendering/test_depth_backprojection_conventions.py oracle_rri/tests/integration/test_render_backproject_real_data.py`

From `oracle_rri/`:

- `.venv/bin/pytest -q tests/rendering/test_depth_backprojection_conventions.py`
- `.venv/bin/pytest -q tests/integration/test_render_backproject_real_data.py`

## Notes / Follow-ups

- The “blank wall gets high RRI” issue is likely metric-related (RRI based only on semi-dense points can
  under-penalize untextured surfaces); it needs an explicit *completeness* / mesh-distance term and is orthogonal to
  the pose/render/backproject convention bugs fixed here.


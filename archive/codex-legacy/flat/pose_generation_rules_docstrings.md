# Task: Refine pose-generation rule docstrings

## Summary of changes
- Clarified and slightly condensed docstrings in `oracle_rri/oracle_rri/pose_generation/candidate_generation_rules.py` for:
  - `ShellSamplingRule`
  - `MinDistanceToMeshRule`
  - `PathCollisionRule`
  - `FreeSpaceRule`
  - `_point_mesh_distance_efm3d`
- Emphasised conceptual behaviour (sampling on a spherical cap, collision handling, free-space AABB) while keeping implementation details accessible to a CS graduate student with CV background.
- Added explicit argument/return type descriptions aligned with project typing conventions.

## Notable implementation details
- `ShellSamplingRule` docstring now explicitly explains:
  - Radius sampling on a thick spherical shell.
  - Area-uniform sampling on a spherical cap for `UNIFORM_SPHERE ` via `u ~ U(sin(theta_min), sin(theta_max))`.
  - Back-transformation from rig frame to VIO world via `PoseTW` and orientation using `view_axes_from_points`.
- `_sample_directions` docstring clarifies shapes, frames, and the intuition behind `FORWARD_POWERSPHERICAL ` vs `UNIFORM_SPHERE `.
- `MinDistanceToMeshRule` and `PathCollisionRule` docstrings now clearly separate CPU (trimesh / PyEmbree) and GPU (EFM3D-style point–triangle distances) paths.
- `FreeSpaceRule` docstring now links the AABB to SLAM / mesh-derived occupancy metadata.
- `_point_mesh_distance_efm3d` docstring emphasises its role as a minimal, dependency-light analogue of PyTorch3D's point–mesh distance.

## Lint / style fixes
- Resolved `ruff` issues introduced or exposed during edits:
  - Used local `verts` / `faces` variables in `PathCollisionRule` when calling `_point_mesh_distance_efm3d`.
  - Adjusted `_edge_dist` helper in `_point_mesh_distance_efm3d` to take `pts_local` explicitly, removing a closure over `pts` and satisfying B023.
- Verified formatting and linting:
  - `ruff format oracle_rri/oracle_rri/pose_generation/candidate_generation_rules.py`
  - `ruff check oracle_rri/oracle_rri/pose_generation/candidate_generation_rules.py` (all checks pass).

## Testing status
- Global `pytest` run (configured `testpaths = oracle_rri/tests`) fails during collection due to a missing module dependency:
  - `ModuleNotFoundError: No module named 'oracle_rri.views'` in `oracle_rri/tests/test_candidate_rendering.py`.
- This error is unrelated to the pose-generation rules module and indicates an incomplete package layout (missing `oracle_rri/views.py` or package).
- No additional failures observed before collection halted; pose-generation changes are limited to docstrings and minor internal helper refactors.

## Potential follow-ups
- Restore or add the `oracle_rri.views` package/module expected by `test_candidate_rendering.py`, or update tests to the current package layout.
- Once `oracle_rri.views` is in place, re-run the full test suite to ensure integration coverage for pose generation and candidate rendering.
- Consider adding a short high-level Quarto documentation snippet summarising the sampling rules, reusing the clarified docstrings.


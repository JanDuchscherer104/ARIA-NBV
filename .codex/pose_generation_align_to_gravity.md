# `align_to_gravity` (pose_generation)

## Goal

Fix the “tilted/skewed torus” issue in candidate position sampling when the reference pose has strong roll/pitch by implementing the existing config flag `CandidateViewGeneratorConfig.align_to_gravity`.

## What was implemented

- Added a gravity-alignment helper in `oracle_rri/oracle_rri/pose_generation/candidate_generation.py`:
  - `_gravity_align_pose(reference_pose)` constructs a **sampling pose** with:
    - **translation unchanged**
    - **up (+Y) aligned to VIO world-up** (`oracle_rri.utils.frames.world_up_tensor`)
    - **yaw preserved** by using the **horizontal projection of the original forward axis** as the new +Z axis
    - (effectively removes pitch + roll in the sampling frame)
- Wired it into `CandidateViewGenerator.generate()`:
  - when `align_to_gravity=True`, `PositionSampler.sample()` and `OrientationBuilder.build()` use the gravity-aligned sampling pose
  - the **returned** `reference_pose` in results remains the physical input reference pose (only the sampling frame is modified)
- Exposed the toggle in the Streamlit sidebar (`oracle_rri/oracle_rri/app/ui.py`).

## Tests added

- `oracle_rri/tests/pose_generation/test_align_to_gravity.py`
  - Verifies world-elevation caps are respected under `align_to_gravity=True` even for a 90° rolled reference pose.
  - Verifies `ViewDirectionMode.FORWARD_RIG` produces gravity-level orientations when `align_to_gravity=True`.

## Docs update

- `docs/contents/todos.qmd` now marks the `align_to_gravity` bullet as implemented.

## Notes / follow-ups

- Current alignment uses the project’s `world_up_tensor()` (constant); if you want per-snippet gravity, consider plumbing `sample.trajectory.gravity_in_world` into candidate generation and using that instead.
- Some pre-existing pose_generation tests are currently broken/unrelated:
  - `oracle_rri/tests/pose_generation/test_pose_generation_revised.py` imports a missing module (`reference_power_spherical_distributions`).
  - `oracle_rri/tests/pose_generation/test_candidate_generation_mesh_access.py` monkeypatches a missing symbol (`mesh_from_snippet`).


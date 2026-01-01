# Pose Generation: Jitter + Angle Diagnostics (Dec 16, 2025)

## Problem recap

The Streamlit “Candidate Poses → Diagnostics” view made it hard to reason about view-direction jitter:

- The “Euler angles” plots showed a **pitch of ~0 for all candidates**, which looked wrong.
- The azimuth/elevation/roll jitter knobs (`view_max_azimuth_deg`, `view_max_elevation_deg`, `view_roll_jitter_deg`)
  appeared to affect the “wrong” plotted dimensions.
- Some plots mixed display-only conventions (`rotate_yaw_cw90`) with physical sampling coordinates, causing further
  confusion (e.g., candidate position torus appearing around an unexpected axis).

## Root causes

1) **Rig-basis mismatch + double-application in plotting**

The incoming `t_world_rig` basis is effectively *twisted* relative to the LUF convention assumed by our samplers
(LUF: x=left, y=up, z=fwd). A fixed 90° rotation about the pose-local +Z/forward axis (`rotate_yaw_cw90`) is needed
as a **rig-basis correction**; without it, azimuth/elevation appear swapped and the candidate shell “torus” aligns with
the wrong reference axis.

Once we apply this correction in the physical pipeline, we must **not apply it again** in candidate plotting.
The candidate frusta plot was still applying `rotate_yaw_cw90` as a “display rotation”, which adds a constant extra
roll offset. This is especially obvious when roll jitter is enabled, because both rotations are about the same axis
(local +Z/forward).

2) **Angle plots did not match the pose-generation parameterization**

- `PoseTW.to_ypr` / `PoseTW.to_euler` use a **ZYX** decomposition in EFM3D and are not the same as our
  **LUF view angles**:
  - view “yaw” = azimuth around +Y (up),
  - view “pitch” = elevation around +X (left),
  - view “roll” = twist around +Z (forward).
- For the roll-free look-at frames used by candidate generation, one component of a ZYX decomposition can be
  identically ~0 by construction, which is expected but looks like a bug if interpreted as “camera pitch”.
- The `view_dirs_delta` stats were extracted with `PoseTW.to_ypr`, which decomposes around different axes than the
  delta construction in `OrientationBuilder` (local yaw/pitch about +Y/+X and roll about +Z).

## What was changed

- Physical pipeline:
  - Applied `rotate_yaw_cw90(reference_pose)` inside `CandidateViewGenerator.generate()` as a **rig-basis correction**
    so candidate positions/orientations are consistent in the world frame.
  - Candidate plots now default to the **physical** convention (`display_rotate=False`) so the frusta/axes are not
    twisted a second time.
- Diagnostics:
  - Replaced “Euler” histograms with **view yaw/pitch/roll** derived from camera forward/up vectors:
    - world: yaw around world-up (+Z), pitch as elevation, roll as twist around forward.
    - reference: yaw/pitch/roll in LUF (x=left, y=up, z=fwd).
  - Updated `view_dirs_delta` stats to report LUF yaw/pitch/roll consistent with the jitter knobs.
- Docs/UI:
  - Updated `view_sampling_strategy` documentation/help text to reflect the current precedence:
    bounded box jitter via `view_max_*` caps takes priority; the sampler is only used when both caps are 0.
- Tests:
  - Updated `tests/pose_generation/test_orientations.py` to match `OrientationBuilder.build()` returning
    `(poses, delta)` and added a roll-jitter regression test.

## Key code locations

- Candidate generation: `oracle_rri/oracle_rri/pose_generation/candidate_generation.py`
- Orientation jitter: `oracle_rri/oracle_rri/pose_generation/orientations.py`
- Diagnostics plots: `oracle_rri/oracle_rri/pose_generation/plotting.py`,
  `oracle_rri/oracle_rri/app/panels.py`
- Streamlit knobs: `oracle_rri/oracle_rri/app/ui.py`

## Follow-ups / suggestions

- Consider exposing a single “display rotation” toggle in the UI that applies consistently across candidate position,
  frusta, and direction plots.
- If we keep both `view_sampling_strategy` and bounded jitter caps, consider renaming the sampler to make its role
  explicit (e.g., “unbounded/legacy view sampler”) and document the precedence in a dedicated conventions page.

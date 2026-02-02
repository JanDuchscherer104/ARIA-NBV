# Paper note: CW90 / `rotate_yaw_cw90` placement

Date: 2026-01-30

## What changed

- Moved the CW90 discussion out of `docs/typst/paper/sections/05-coordinate-conventions.typ` into the appendix section `@sec:appendix-pose-frames`.
- Added an explicit explanation that `rotate_yaw_cw90` is a 90° twist about the pose-local `+Z` (forward) axis (historical name), and that candidate sampling is performed around the rotated reference pose (#symb.ase.traj_final) to avoid an azimuth/elevation swap under the LUF sampling assumption.
- Kept the coordinate conventions section as a short pointer to the appendix to avoid future drift.

## Consistency reminder

- If `rotate_yaw_cw90` is undone for learning/diagnostics, the same correction must be applied to poses and the associated `p3d_cameras` to keep pose encoding and projection features aligned.


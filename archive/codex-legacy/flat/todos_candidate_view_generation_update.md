# Update `docs/contents/todos.qmd` (Candidate View Generation & Sampling)

## What changed

Rewrote the "Candidate View Generation & Sampling" section to reflect the current `oracle_rri/oracle_rri/pose_generation` implementation:

- Added a concise description of the current 3-stage pipeline:
  - `PositionSampler` → `OrientationBuilder` → pruning rules (`FreeSpaceRule`, `MinDistanceToMeshRule`, `PathCollisionRule`).
- Kept the original issue statements + figure(s), and added **root cause + fix** notes for each.
- Introduced a dedicated “Implementation checklist (non-issues)” where all bullets are checkboxes and marked `[x]` when done.

## Open follow-ups captured as TODOs

- Interpret roll/pitch/yaw histograms across frames (world vs reference vs display-rotated).
- Revisit the “pitch histogram” question once we standardize the frame used for reporting.


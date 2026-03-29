# Slide–Paper Sync Report: VIN Scoring Architecture

## Scope
- Reviewed `docs/typst/slides/slides_4.typ` from “VIN Scoring Architecture” up to (but not including) “VINv3 forward: feature branches”.
- Treated slides as ground truth and aligned the paper accordingly.

## Findings
- The composite VIN-input figure `vin-geom-oc_pr-candfrusta-semi-dense.png` was used in slides but missing from the paper.
- The paper did not explicitly state that EVL evidence channels (`occ_in`, `counts`) are derived from semi-dense SLAM points, nor the default voxel grid size used in the current EVL config (48³ ~ 4 m).

## Changes Applied
- Added the VIN input composite figure to the architecture section and to the appendix slide-figure collection.
- Updated the core-architecture bullet to mention the optional projection-grid CNN.
- Added a concise sentence about voxel grid size and semi-dense provenance of observation-evidence channels.

## Files Touched
- `docs/typst/paper/sections/06-architecture.typ`
- `docs/typst/paper/sections/12i-appendix-slide-figures.typ`

## Follow-ups
- None in this slice. Next steps begin at “VINv3 forward: feature branches”.

## Tests / Builds
- No tests or renders run (documentation-only changes).

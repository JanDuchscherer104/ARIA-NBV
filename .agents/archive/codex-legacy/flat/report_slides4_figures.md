# Slides 4 Figure Inclusion Report (2026-01-29)

## Scope
- Updated the “Design intent” slide with clearer contract and motivation.
- Added all missing `app-paper` figures to `docs/typst/slides/slides_4.typ` without removing any existing figures.

## Figures Added (new placements)
- `field_occ_in.png`, `field_occ_pr.png`, `field_counts_norm.png` on new slide “EVL field slices (examples)”.
- `semi-dense-pc-cand-vis.png` added to “Backprojection” slide next to `backproj+semi.png`.
- `field_cent_pr_nms_bug.png` added on new diagnostics slide “Failure case: cent_pr_nms artifact”.

## Notes
- Existing figures remain unchanged; all list items are now referenced in `slides_4.typ`.
- Layout uses `#figure(...)` + `#grid(...)` per Typst conventions.

## Follow-ups
- Optional: render slides to check spacing after the new multi-panel grids.

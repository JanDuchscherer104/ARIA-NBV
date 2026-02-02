# Slide figures added to paper appendix (2026-01-29)

## Summary
- Added a new appendix section that includes every figure used in `docs/typst/slides/slides_4.typ`
  that was previously missing from the paper.
- Grouped figures into themed grids (app-paper, offline cache, VIN visuals, Optuna, W&B, logo).

## Files
- `docs/typst/paper/sections/12i-appendix-slide-figures.typ`
- `docs/typst/paper/main.typ`

## Compile
- `typst compile docs/typst/paper/main.typ /tmp/paper.pdf --root docs`

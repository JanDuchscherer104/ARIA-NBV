# Paper appendix: OracleRriLabeler pipeline (2026-01-03)

## What changed
- Added a new appendix section describing the end-to-end oracle label pipeline (candidate generation → depth rendering → backprojection → oracle RRI → ordinal binning).
- Included code-faithful equations and implementation notes, including why backprojection uses NDC for consistency with PyTorch3D rasterization.
- Added a citation for the Power Spherical distribution paper.

## Files
- `docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ`
- `docs/typst/paper/main.typ`
- `docs/typst/shared/macros.typ`
- `docs/references.bib`

## Notes / follow-ups
- If we expand the appendix further, consider adding a small table mapping pipeline stages to concrete entry points and tensor shapes.
- Consider adding an explicit citation for the PyTorch3D paper (not just docs) if needed for publication.

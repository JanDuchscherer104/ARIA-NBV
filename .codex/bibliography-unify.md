## Summary
- Unified bibliography to a single source of truth: `docs/references.bib`.

## Changes
- Updated Typst paper to use `/references.bib` (rooted at `docs`).
- Updated `docs/uml_diagrams/_quarto.yml` to point at `../references.bib`.
- Removed duplicate bib files:
  - `docs/_shared/references.bib`
  - `docs/typst/references.bib`
  - `docs/typst/paper/references.bib`

## Notes
- All bib files had identical 49 entries; no citation loss.
- Typst builds should be run with root `docs` so `/references.bib` resolves.

# Slide ↔ Paper Figure Alignment (slides_4)

## Goal
Align paper figures with the slide deck: keep only slide figures in the paper (plus EFM3D/EVL arch), remove paper-only figures, and improve appendix integration.

## Key changes
- Removed paper-only figures across dataset, diagnostics, evaluation, and appendix gallery sections.
- Replaced oracle RRI figures with slide equivalents (cand_renders_1x3, depth_histograms_3x3, backproj+semi, semidense vis, acc/comp, oracle_rri_bar).
- Reworked the appendix slide gallery (`12i`) to include all slide figures grouped by topic.
- Simplified the old appendix gallery (`12-appendix-gallery`) to avoid non-slide figures.
- Updated architecture section to use EVL output summary and scene-field slices from slides.

## Files touched
- `docs/typst/paper/sections/04-dataset.typ`
- `docs/typst/paper/sections/05-oracle-rri.typ`
- `docs/typst/paper/sections/06-architecture.typ`
- `docs/typst/paper/sections/08-system-pipeline.typ`
- `docs/typst/paper/sections/09-diagnostics.typ`
- `docs/typst/paper/sections/09a-evaluation.typ`
- `docs/typst/paper/sections/12-appendix-gallery.typ`
- `docs/typst/paper/sections/12b-appendix-extra.typ`
- `docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ`
- `docs/typst/paper/sections/12i-appendix-slide-figures.typ`

## Notes
- Duplicate figure labels were avoided by removing labels from appendix duplicates.
- Only EFM3D/EVL arch figure remains outside slides as requested.

## Follow-ups
- If desired, further reduce duplicate figures by removing slide duplicates in main sections.
- Run Typst build to ensure no duplicate labels or missing references.

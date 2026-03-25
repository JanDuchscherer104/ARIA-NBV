---
id: 2026-01-29_slide_figures_appendix_2026-01-29
date: 2026-01-29
title: "Slide Figures Appendix 2026 01 29"
status: legacy-imported
topics: [slide, figures, appendix, 2026, 01]
source_legacy_path: ".codex/slide_figures_appendix_2026-01-29.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

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

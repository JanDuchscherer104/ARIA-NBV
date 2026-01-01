# KaTeX math rendering switch (2025-12-24)

## Change

Set Quarto HTML math rendering to KaTeX:

- `docs/_quarto.yml`: `format: html: html-math-method: katex`

## Why

MathJax default math fonts (and especially nested subscripts) looked visually inconsistent with the site’s body font.
Switching to KaTeX changes the math font rendering pipeline.

## Validation

- Rendered `docs/contents/impl/vin_nbv.qmd` successfully with `html-math-method: katex`.

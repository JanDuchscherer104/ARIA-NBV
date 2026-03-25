---
id: 2025-12-24_katex_math_method_2025-12-24
date: 2025-12-24
title: "Katex Math Method 2025 12 24"
status: legacy-imported
topics: [katex, math, method, 2025, 12]
source_legacy_path: ".codex/katex_math_method_2025-12-24.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# KaTeX math rendering switch (2025-12-24)

## Change

Set Quarto HTML math rendering to KaTeX:

- `docs/_quarto.yml`: `format: html: html-math-method: katex`

## Why

MathJax default math fonts (and especially nested subscripts) looked visually inconsistent with the site’s body font.
Switching to KaTeX changes the math font rendering pipeline.

## Validation

- Rendered `docs/contents/impl/vin_nbv.qmd` successfully with `html-math-method: katex`.

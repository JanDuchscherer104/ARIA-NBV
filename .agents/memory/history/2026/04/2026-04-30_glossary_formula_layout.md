---
id: 2026-04-30_glossary_formula_layout
date: 2026-04-30
title: "Glossary Formula Rendering And Full-Width Layout"
status: done
topics: [docs, glossary, quarto, kg]
confidence: high
canonical_updates_needed: []
files_touched:
  - scripts/glossary_build.py
  - docs/glossary/terms.yml
  - docs/contents/glossary.qmd
  - docs/styles.css
  - docs/_generated/context/glossary.jsonl
---

## Task

Fix the generated glossary so formulas render as Quarto math instead of escaped text, add the missing target-specific RRI equation, and make the glossary use the wider public docs layout.

## Method

Updated the glossary generator to emit formula blocks as fenced Quarto divs containing raw display math. The generator now accepts the existing `formula.tex` shape and a future `formulae` list for multi-equation terms. The glossary page front matter now sets `page-layout: full` at the top level and under HTML format metadata so the current Quarto build emits `page-layout-full`.

## Findings

The earlier broken formulas came from putting escaped TeX inside raw HTML. Pandoc does not parse that content as math, which left reader-facing text such as `[ (q)= ]`. Rendering the formulas as Markdown display math inside the generated card markup lets Quarto and MathJax process the equations correctly.

## Verification

Ran `make glossary` and `cd docs && quarto render contents/glossary.qmd`. The rendered HTML contains math display spans for scene-level RRI, target-specific RRI, and Track, and the live page inspection showed MathJax containers for all three formulas with no broken formula text in `document.body.innerText`.

Live layout inspection at 1440 px showed the metrics card grid using two columns in the full-width layout. A 390 px mobile emulation showed `documentElement.scrollWidth == innerWidth`, so the glossary has no horizontal overflow in that viewport.

## Canonical State Impact

No additional project-state memory update is needed. This is a public docs rendering and generated artifact change; the canonical glossary source is now `docs/typst/shared/glossary.typ`.

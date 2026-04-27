---
id: 2026-03-31_hestia_literature_review
date: 2026-03-31
title: "Add Hestia literature review page"
status: done
topics: [docs, literature, quarto, hestia, nbv]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/literature/hesita.qmd
  - docs/contents/literature/index.qmd
  - docs/references.bib
artifacts:
  - docs/_site/contents/literature/hesita.html
  - docs/_site/contents/literature/index.html
---

## Task

Add a new Quarto literature review for Hestia that is grounded in the local
LaTeX source, aligned with the current paper and ideas scratchpad, and written
in the style of the existing literature pages.

## Method

Read `docs/typst/paper/main.typ` plus the key paper sections on introduction,
related work, architecture, and future extensions. Read `docs/contents/ideas.qmd`,
the existing literature pages under `docs/contents/literature/`, the seeded
ChatGPT note at `.agents/tmp/ChatGPT-reports/hesita-literature-review.md`, and
the local Hestia manuscript under `docs/literature/tex-src/arXiv-hesita/`.
Rewrote `hesita.qmd` from those sources rather than copying the seed note,
added a bibliography entry, and linked the page from the literature index.

## Findings

- Hestia is most relevant to Aria-NBV as a control and representation pattern,
  not as a replacement for the RRI objective.
- The most transferable ideas are directional face visibility, hierarchical
  look-at-then-position control, supervised intermediate targets, and
  collision-feasibility projection.
- The local source reports the quantitative claims needed for the page:
  at least +4 CR, nearly 50% lower CD, 25 FPS, strong limited-budget gains,
  and robustness from larger Objaverse-based training.

## Verification

- `cd docs && quarto check`
- `cd docs && quarto render contents/literature/hesita.qmd`
- `cd docs && quarto render contents/literature/index.qmd`

## Canonical State Impact

No canonical state files changed. This was a documentation and literature
context update.

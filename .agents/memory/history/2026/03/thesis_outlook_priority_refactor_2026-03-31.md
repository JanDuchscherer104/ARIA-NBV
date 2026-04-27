---
id: 2026-03-31_thesis_outlook_priority_refactor
date: 2026-03-31
title: "Thesis Outlook Priority Refactor"
status: done
topics: [slides, typst, thesis, advisor]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - docs/typst/slides/slides_thesis_outlook.pdf
---

## Task
Refactor the advisor-facing thesis outlook deck so it reflects the updated canonical priorities more cleanly and uses shared symbols / equations from `docs/typst/shared/macros.typ`.

## Method
Reworked the slide order and content around the current canonical project priorities: compressed the early decision slides into a top-5 agenda plus recommended answers, added an explicit blockers slide, moved the theory / MDP contract ahead of implementation evidence, merged the implementation material into one evidence slide, pushed simulator detail into backup, and updated the math slide to use the shared `symb` / `eqs` notation surface.

## Findings
The strongest advisor-facing structure is a short decision memo, not an implementation walkthrough. The resulting deck now foregrounds scope lock, current blockers, Hestia takeaways, the geometry-first MDP contract, and the evidence boundary in that order.

## Verification
- `typst compile docs/typst/slides/slides_thesis_outlook.typ docs/typst/slides/slides_thesis_outlook.pdf --root docs`
- visual inspection via `pdftoppm` and page/contact-sheet review

## Canonical State Impact
No canonical state docs changed. The slide deck was brought into closer alignment with the existing canonical priorities and decision order.

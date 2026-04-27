---
id: 2026-03-30_thesis_outlook_slide_refinement
date: 2026-03-30
title: "Thesis Outlook Slide Refinement"
status: done
topics: [slides, typst, thesis, rl, counterfactuals]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - docs/typst/slides/slides_thesis_outlook.pdf
  - docs/figures/thesis_outlook/counterfactual_paths.png
  - docs/figures/thesis_outlook/counterfactual_step.png
  - docs/figures/thesis_outlook/policy_comparison.png
---

Task: refine the advisor-facing thesis outlook deck using the highest-signal current truth from `docs/contents/ideas.qmd`, canonical state memory, and the paper discussion/extensions sections.

Method: restructured the slide narrative around decisions, recommended thesis scope, implemented evidence, geometry-first RL formulation, unresolved questions, and concrete milestones. Ran iterative compile-and-inspect loops with `typst compile`, `pdfinfo`, `pdftoppm`, and visual preview checks to remove automatic slide splits and trim dense theory content into meeting-friendly panels.

Findings / outputs:
- the previous draft had advisor-relevant content but overflowed into 14 pages because several three-block theory slides split automatically
- reducing the deck to two-block decision/theory slides preserved the important content while returning the deck to 10 pages
- `ideas.qmd` was the best source for high-priority professor-facing decisions: stay geometry-first on ASE mesh-backed supervision, prioritize planning + discrete shell RL over a large VIN rewrite, de-scope continuous PPO-first work, and surface workstation / ASE simulator access as leverage items
- synthetic rollout / RL figures remain useful as implementation evidence, but are explicitly labeled as diagnostics rather than ASE results

Verification:
- `cd docs && typst compile typst/slides/slides_thesis_outlook.typ typst/slides/slides_thesis_outlook.pdf --root .`
- `pdfinfo docs/typst/slides/slides_thesis_outlook.pdf`
- `pdftoppm -png docs/typst/slides/slides_thesis_outlook.pdf /tmp/thesis_refined_pages3/page`
- visual inspection of contact sheet and fresh per-page renders

Canonical state impact: none. This was a presentation refinement pass aligned to existing project state rather than a change to project truth.

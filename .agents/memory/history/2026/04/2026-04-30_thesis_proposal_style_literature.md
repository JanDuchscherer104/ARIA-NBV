---
id: 2026-04-30_thesis_proposal_style_literature
date: 2026-04-30
title: "Thesis Proposal Style and Literature Expansion"
status: done
topics: [docs, thesis, typst, literature, nbv, rl]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/proposal.typ
  - docs/typst/thesis/proposal.pdf
  - docs/typst/thesis/sections/proposal/_style.typ
  - docs/typst/thesis/sections/proposal/01-motivation.typ
  - docs/typst/thesis/sections/proposal/02-related-work.typ
  - docs/typst/thesis/sections/proposal/02-problem.typ
  - docs/typst/thesis/sections/proposal/04-method.typ
  - docs/typst/thesis/sections/proposal/06-outline.typ
  - docs/contents/literature/index.qmd
  - docs/contents/literature/active_3dgs_nbv.qmd
  - docs/contents/literature/rl_planning.qmd
  - docs/references.bib
---

## Task

The user asked to improve the Typst proposal's visual style, consider Typst package options, broaden the literature review, and add more links to important sources.

## Method

I followed the Typst and docs workflows: checked local template constraints, consulted Typst table/box/headings guidance, reviewed current thesis roadmap/questions and literature pages, and browsed current Typst Universe/literature source pages for package and citation details. I chose a local built-in Typst style layer instead of adding a hard package dependency; `booktabs` compiled locally, but built-in tables/boxes were sufficient and more portable.

## Outputs

The proposal now has a reusable proposal-local style file with colored section headings, compact inline raw styling, lighter tables, a thesis-position callout, and a visual `ArgTopK -> ArgTop1_h` rollout ladder. A new related-work section positions ARIA-NBV against active perception, classical view planning, continuous NBV policies, radiance-field/3DGS view selection, and offline/stochastic planning. The literature index and relevant literature pages now link to the expanded source set, and `docs/references.bib` contains primary bibliography entries for active perception, view planning, efficient NBV, NeRF/3DGS, ActiveNeRF/FisherRF, PPO, and SAC.

## Verification

`cd docs && typst compile typst/thesis/proposal.typ --root .` succeeds. The proposal PDF renders to 18 pages and was visually inspected as PNG pages, including the new callout, literature table, rollout ladder, and schedule tables. `make qmd-frontmatter-check` succeeds. The edited literature pages render individually with Quarto. A single combined `quarto render` invocation failed because Quarto treated multiple files as a combined render target; rerunning each page separately succeeded.

## Canonical State Impact

No canonical project-memory update is needed. The changes refine proposal presentation and literature links without changing the locked thesis decisions: bounded oracle-RRI rollout remains the first non-myopic comparison, and continuous RL remains behind the evidence gate.

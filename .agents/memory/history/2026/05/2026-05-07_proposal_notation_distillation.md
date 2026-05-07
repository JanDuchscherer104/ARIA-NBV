---
id: 2026-05-07_proposal_notation_distillation
date: 2026-05-07
title: "Proposal Notation Distillation"
status: done
topics: [thesis, proposal, typst, rri, notation]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/thesis/roadmap.qmd
  - docs/typst/thesis/proposal.pdf
  - docs/typst/thesis/sections/proposal/02-problem.typ
  - docs/typst/thesis/sections/proposal/03-objectives.typ
  - docs/typst/thesis/sections/proposal/04-method.typ
artifacts:
  - docs/typst/thesis/proposal.pdf
---

## Task

Fix proposal notation after visual review: avoid Typst subscript/function
parsing issues, stop presenting the oracle distance as generic `CD`, and
distill noisy proposal sections without losing the target-conditioned
finite-candidate `Q_H` thesis signal.

## Method

Rewrote the proposal problem, objectives, and method sections into shorter
forms. Replaced `D_e(P)` / `CD(...)` style notation with `Delta_t^e`, defined
as point-mesh accuracy plus mesh-to-point completeness, matching the implemented
`pm_acc_*` / `pm_comp_*` diagnostics. Replaced fragile token equations such as
`x_scene(s_t)` with compact scene/target/history/candidate token notation.
Aligned the roadmap model to the same `Delta_t^e` point-mesh notation.

## Verification

- `cd docs && typst compile typst/thesis/proposal.typ --root . /tmp/proposal-distilled.pdf`
- `make proposal-pdf`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/thesis/roadmap.qmd && quarto render contents/thesis/questions.qmd`
- Visual spot-check of rendered proposal pages around the problem equations.
- `rg` check for removed fragile patterns: `CD`, `D_e(`, `C_e(`, `x_"..."`, shared `CD` equation includes.
- Three litkg claim checks for thesis core, V1 actor/oracle boundary, and `Q_H` success bar returned `supported` with confidence `1.0`.

## Canonical State Impact

No canonical memory update is needed. This was a notation and presentation
distillation, not a thesis-contract change.

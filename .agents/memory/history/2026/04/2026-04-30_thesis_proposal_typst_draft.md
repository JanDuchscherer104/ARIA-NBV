---
id: 2026-04-30_thesis_proposal_typst_draft
date: 2026-04-30
title: "Thesis Proposal Typst Draft"
status: done
topics: [docs, thesis, typst, nbv, rri, planning]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/proposal.typ
  - docs/typst/thesis/sections/proposal/01-motivation.typ
  - docs/typst/thesis/sections/proposal/02-problem.typ
  - docs/typst/thesis/sections/proposal/03-objectives.typ
  - docs/typst/thesis/sections/proposal/04-method.typ
  - docs/typst/thesis/sections/proposal/05-schedule.typ
  - docs/typst/thesis/sections/proposal/06-outline.typ
  - docs/typst/thesis/proposal.pdf
---

## Task

Draft the Typst thesis proposal from the current ARIA-NBV implementation, roadmap, research questions, and local Semantic Scholar / litkg literature context.

## Method

Grounded the proposal in the seminar paper, canonical project memory, thesis roadmap, research questions, RL planning theory page, local literature review pages, and the litkg context pack. The proposal preserves the current thesis boundary: target-conditioned, quality-driven NBV with target-specific RRI and bounded rollout evaluation, while treating continuous RL as evidence-gated future work.

## Outputs

The proposal wrapper now supplies a proposal-specific German title, proposal submission date, and AI-transparency statement. The six proposal sections now contain full prose plus compact objective, schedule, and outline tables. The rendered PDF was regenerated.

## Verification

Validated with `cd docs && typst compile typst/thesis/proposal.typ --root .`, PDF text checks for visible TODO/citation artifacts, spot visual inspection of rendered pages, `git diff --check`, `make context`, and `make check-agent-memory`.

## Canonical State Impact

No canonical state update was needed. The proposal is a public thesis writing artifact aligned to the current roadmap and memory state.

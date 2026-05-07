---
id: 2026-05-07_proposal_dense_scientific_rewrite
date: 2026-05-07
title: "Proposal Dense Scientific Rewrite"
status: done
topics: [thesis, proposal, typst, literature, litkg]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/proposal.typ
  - docs/typst/thesis/proposal.pdf
  - docs/typst/thesis/sections/proposal/01-motivation.typ
  - docs/typst/thesis/sections/proposal/02-problem.typ
  - docs/typst/thesis/sections/proposal/02-related-work.typ
  - docs/typst/thesis/sections/proposal/03-objectives.typ
  - docs/typst/thesis/sections/proposal/04-method.typ
  - docs/typst/thesis/sections/proposal/05-schedule.typ
  - docs/typst/thesis/sections/proposal/06-outline.typ
  - docs/typst/thesis/figures/proposal_system_flow.mmd
  - docs/typst/thesis/figures/proposal_system_flow.png
  - docs/typst/thesis/figures/proposal_gantt.mmd
  - docs/typst/thesis/figures/proposal_gantt.png
artifacts:
  - docs/typst/thesis/proposal.pdf
---

## Task

Rewrite the thesis proposal into a denser, more technical, and more coherent Typst document grounded in `questions.qmd`, `ideas.qmd`, `roadmap.qmd`, `docs/literature/sources.jsonl`, `docs/references.bib`, local thesis memory, and litkg context.

## Method

Applied `agent-behavior`, `docs-curator`, `aria-litkg-memory`, `plan-grill`, and `typst-authoring` guidance. Used litkg capability, route, search, and claim-check commands to ground the scope around target-conditioned, quality-driven finite-candidate NBV on ASE/EFM, with `Q_H` as the mandatory M5 result and continuous control/simulator/SceneScript/real-device work framed as bridge or future work.

The proposal now includes all section files from `proposal.typ`, formalizes the finite-candidate target-conditioned problem, adds candidate-query Transformer and rollout equations, expands related work across the local paper registry and bibliography, and embeds Mermaid source diagrams rendered as PNG figures for Typst compatibility.

## Outputs

The generated Mermaid diagrams are stored as source `.mmd` files plus rendered PNGs under `docs/typst/thesis/figures/`. Rendering used Mermaid CLI with system Chrome via `PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome` because the local Puppeteer cache did not contain Chrome.

The rewritten proposal keeps the litkg warning explicit: ARIA-NBV is not framed as a finished deployed RL policy. The central claim is a planned, bounded-horizon `Q_H` policy-like finite-candidate scorer trained from ASE oracle rollouts and evaluated against one-step and oracle lookahead baselines.

## Verification

- `cd docs && typst compile typst/thesis/proposal.typ --root . /tmp/proposal-revised.pdf`
- `make proposal-pdf`
- `make qmd-frontmatter-check`
- `make kg-claim-check KG_FORMAT=json KG_CLAIM='ARIA-NBV thesis core is target-conditioned, quality-driven finite-candidate NBV on ASE/EFM using target-specific RRI, with continuous control, simulators, SceneScript, and real-device work treated as bridge or future work.'`
- `make kg-claim-check KG_FORMAT=json KG_CLAIM='V1 OBS-SEL / PRED-Q / GT-EVAL uses observed or predicted target descriptors for actor input while GT target crops and GT boxes are used only for oracle labels and evaluation.'`
- `make kg-claim-check KG_FORMAT=json KG_CLAIM='The mandatory M5 result is a target-conditioned candidate-query Transformer Q_H trained from ASE oracle rollout data that must beat one-step greedy or model scoring on cumulative target RRI under equal acquisition budget, with oracle lookahead reported as an upper bound.'`

All three claim checks returned `supported` with confidence `1.0`. The only relevant risk flag was the dirty `.agents` worktree/backlog warning, which predated this proposal work and was not treated as final backlog truth.

## Canonical State Impact

No canonical memory update is needed. The rewrite follows existing canonical thesis direction rather than changing it.

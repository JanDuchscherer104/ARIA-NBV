---
id: 2026-05-08_proposal_distillation_headroom
date: 2026-05-08
title: "Proposal Distillation And Headroom-Gated Roadmap"
status: done
topics: [thesis, proposal, roadmap, typst, quarto, litkg]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/proposal.pdf
  - docs/typst/thesis/sections/proposal/
  - docs/typst/thesis/figures/proposal_system_flow.mmd
  - docs/typst/thesis/figures/proposal_system_flow.png
  - docs/contents/thesis/roadmap.qmd
  - docs/contents/thesis/questions.qmd
  - docs/references.bib
---

## Task

Distill the advisor-facing Typst proposal into a compact scientific research
contract and move longer planning detail, including the Gantt, into the roadmap.

## Method

The proposal was rewritten around three aims: leakage-safe target-RRI oracle,
target-conditioned one-step scoring, and headroom-gated finite-candidate
$Q_H$. The Gantt figure and proposal-specific Gantt assets were removed, while
the roadmap Mermaid Gantt was updated with clearer M5 headroom/$Q_H$ wording.
Deep Sets and Set Transformer bibliography keys were added because the proposal
now cites them explicitly.

## Verification

Validated and regenerated the changed Mermaid proposal flow. Rendered the
proposal with Typst, regenerated the tracked proposal PDF, rendered the roadmap
and questions with Quarto, ran the QMD frontmatter check, and ran litkg claim
checks for implemented-versus-planned scope, V1 leakage boundaries, and the
headroom-gated $Q_H$ evaluation claim.

## Canonical State Impact

No canonical memory update is needed. The change is a public narrative
distillation aligned with existing thesis state; it does not change code or data
contracts.

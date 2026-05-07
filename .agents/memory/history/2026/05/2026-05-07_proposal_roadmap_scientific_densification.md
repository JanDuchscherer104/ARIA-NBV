---
id: 2026-05-07_proposal_roadmap_scientific_densification
date: 2026-05-07
title: "Proposal Roadmap Scientific Densification"
status: done
topics: [thesis, proposal, roadmap, typst, quarto, litkg]
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

Implement the accepted proposal + roadmap densification plan by mining
`.agents/work/aria_nbv_review_outputs/`,
`.agents/work/aria_nbv_dense_review_roadmap.typ`, and
`.agents/work/aria_nbv_proposal_dense_sections.typ` without treating those
internal review artifacts as public source-of-truth.

## Method

Kept current thesis ownership with `docs/contents/thesis/questions.qmd`,
`docs/contents/thesis/roadmap.qmd`, `.agents/memory/state/`, and
`docs/typst/thesis/proposal.typ`. The Typst proposal was strengthened around
actor-visible versus oracle-only state, V1 target leakage control, target RRI
reward/return/endpoint metrics, candidate provenance, rollout replay rows, and
masked candidate-query `Q_H` value learning. The roadmap now carries the longer
scientific scaffolding: literature role matrix, mathematical model, Mermaid
flow/Gantt diagrams, ablation matrix, evidence reporting contract, and risk
register.

## Verification

- `cd docs && typst compile typst/thesis/proposal.typ --root . /tmp/proposal-dense.pdf`
- `make proposal-pdf`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/thesis/roadmap.qmd && quarto render contents/thesis/questions.qmd`
- Mermaid CLI validation for both new roadmap Mermaid blocks using system Chrome.
- `scripts/nbv_qmd_outline.sh --compact`
- `make kg-claim-check KG_FORMAT=json KG_CLAIM='ARIA-NBV thesis core is target-conditioned finite-candidate NBV on ASE/EFM with target-specific RRI and Q_H as the mandatory M5 result.'`
- `make kg-claim-check KG_FORMAT=json KG_CLAIM='V1 OBS-SEL / PRED-Q / GT-EVAL uses actor-visible observed or predicted target descriptors for target selection and model input while GT target crops and GT boxes are used only for oracle labels and evaluation.'`
- `make kg-claim-check KG_FORMAT=json KG_CLAIM='Q_H must beat one-step greedy or one-step model scoring on cumulative target RRI under equal acquisition budget after oracle re-evaluation, with bounded oracle lookahead reported as an upper bound.'`

The three split litkg claim checks returned `supported` with confidence `1.0`.
An initial combined claim was intentionally not used as final evidence because
it was too broad and returned `unverifiable`; the split checks match the actual
advisor-facing assertions.

## Canonical State Impact

No canonical memory update is needed. The edit follows existing canonical
direction and does not change the thesis contract.

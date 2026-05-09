---
id: 2026-05-09_advisor_distillation_self_containment
date: 2026-05-09
title: "Advisor Distillation Self-Containment Pass"
status: done
topics: [thesis, typst, advisor-facing, notation, q-h]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
  - docs/typst/shared/equations/entity.typ
---

## Task

Made the advisor distillation handout self-contained without replacing the
full proposal. The pass added explicit target matching notation, V1
actor/oracle leakage boundaries, evidence-reporting contracts, and the R6D
pose versus `S^2` directional-memory architecture distinction.

## Method

Added shared target equations for the actor-visible target descriptor, target
matching score, selected GT target, and acceptance rule. The handout now states
that unmatched or ambiguous targets are target-invalid protocol cases rather
than low target-RRI examples. It also clarifies that masks apply before action
selection, softmax sampling, loss targets, and bootstrap maximization.

Updated the architecture section to separate R6D candidate pose encoding from
actor-visible directional memory on `S^2`. The first directional-memory default
is the moment summary; low-order spherical harmonics remain the richer ablation.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-distillation-self-contained.pdf --root .`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-self-contained-pages --root docs --pages 4-9 --ppi 220`
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ docs/typst/shared/equations/entity.typ docs/typst/shared/equations/features.typ`
- `make kg-claim-check` supported the V1 OBS-SEL / PRED-Q / GT-EVAL actor/GT boundary.
- `make kg-claim-check` supported the planned-not-implemented R6D and `S^2` feature split.
- `make kg-claim-check` supported the headroom-gated `Q_H` interpretation.

Initial claim checks that named the new handout directly returned
`unverifiable` because the KG has not indexed that source yet; equivalent
source-agnostic thesis claims returned `supported`.

## Canonical State Impact

No canonical memory update is needed. The change documents existing thesis
direction and advisor-facing notation rather than changing the research scope.

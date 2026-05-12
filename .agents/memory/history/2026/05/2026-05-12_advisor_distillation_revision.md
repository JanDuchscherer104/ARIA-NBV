---
id: 2026-05-12_advisor_distillation_revision
date: 2026-05-12
title: "Advisor Distillation Revision"
status: done
topics: [thesis, typst, advisor-facing, q-h, notation]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
  - docs/typst/shared/equations/entity.typ
  - docs/typst/shared/equations/features.typ
  - docs/typst/shared/equations/rl.typ
  - docs/references.bib
---

## Task

Revised the advisor distillation handout into a denser research memo centered
on the current ARIA-NBV substrate, the leakage-safe target-RRI thesis claim,
six research questions, the planned value-model architecture, evaluation
contract, adopted literature roles, and roadmap.

## Method

Removed generic NBV background and threshold clutter from the handout. Added
shared reusable equations for target matching, feature construction,
target-conditioned residual `Q_H`, the CORAL interface, dueling residual heads,
and the Hestia-style target/look-at then pose factorization. Replaced the
remaining finite-action notation with `cal(A)_t`, kept accumulated visibility
as actor-visible `S^2` directional memory rather than R6D pose encoding, and
added a Dueling DQN bibliography entry for the planned dueling residual head.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-distillation-revised.pdf --root .`
- `cd docs && typst compile typst/thesis/proposal.typ /tmp/proposal-shared-regression.pdf --root .`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-revised-pages --root docs --pages 1-12 --ppi 220`
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ docs/typst/shared`
- `make kg-claim-check` supported the V1 actor-visible/GT-only leakage boundary.
- `make kg-claim-check` supported the Hestia transfer as bridge inspiration rather than replacement supervision.
- `make kg-claim-check` supported the headroom-gated `Q_H` interpretation.
- `make kg-claim-check` supported that `Q_H` is not implemented yet and still depends on rollout data and storage contracts.

The claim check naming the newly introduced residual-dueling handout detail
returned `unverifiable` because the KG has not indexed the new source yet.
Fallback search and direct canonical-source inspection confirm the safe
wording: `Q_H` is still planned, not implemented; the residual dueling head is
a proposed first architecture detail pending advisor agreement.

## Canonical State Impact

No canonical memory update is needed in this pass. The handout exposes advisor
decisions, especially the CORAL-to-`Q_H` interface and target-match acceptance
protocol, without locking new thesis scope beyond the existing finite-candidate
`Q_H` direction.

---
id: 2026-05-13_advisor_distillation_review_remediation
date: 2026-05-13
title: "Advisor Distillation Review Remediation"
status: done
topics: [typst, advisor-distillation, qh, coral, literature]
confidence: medium
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
  - docs/typst/shared/equations/entity.typ
  - docs/typst/shared/equations/rl.typ
  - docs/references.bib
---

## Task

Applied the advisor-handout review remediation while keeping the scope to the handout, shared equations directly used by it, and bibliography entries needed by newly cited source roles.

## Method

Strengthened the conditional thesis success rule, reframed the experiment as a masked finite-horizon candidate-decision process, added symbolic target-match acceptance filters, fixed the CORAL interface from softmax class probabilities to threshold probabilities plus marginal decoding, and made residual dueling `Q_H` the canonical displayed value definition. The value-model section now marks MLP/DeepSets/Set Transformer controls, QCNet-style RPE as an interaction ablation, `S^2` moment memory as the default directional-memory signal, and SH/histogram memory as richer ablations.

The literature ledger now treats SCONE and MACARONS as coverage/online contrast sources and points Hestia to the WACV 2026 open-access source. External source checks used arXiv pages for SCONE and MACARONS plus the WACV 2026 Hestia PDF.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-distillation-review-remediation.pdf --root .`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-review-remediation-pages --root docs --pages 1-13 --ppi 220`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-review-remediation-pages-more --root docs --pages 14-18 --ppi 220`
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ docs/typst/shared`
- `rg -n "TODO|FIXME|\\?\\?\\?" docs/typst/thesis/advisor_distillation.typ docs/typst/shared`

KG claim checks were attempted for CORAL decoding, QCNet ablation scope, and SCONE/MACARONS contrast scope. All returned `unverifiable` because the current claim-check path reported `paper:*` nodes without source paths. This was treated as a KG limitation rather than a contradiction; local code/docs and external primary pages supported the wording used in the handout.

## Canonical State Impact

No canonical state update is required. The changes refine the advisor-facing handout and reusable equations without changing the locked thesis spine.

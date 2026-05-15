---
id: 2026-05-14_advisor_distillation_literature_review_revision
date: 2026-05-14
title: "Advisor Distillation Literature-Review Revision"
status: done
topics: [thesis, typst, advisor-handout, literature-review, q-h]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
---

## Task

Implemented the advisor-handout literature-review revision for `docs/typst/thesis/advisor_distillation.typ`. The requested direction was to preserve the six-RQ structure while improving presentation hierarchy, claim-site citation support, the geometric-ML rationale, and the evaluation/support ordering.

## Method

Kept the edit scoped to the handout source. Added a front-loaded thesis contract that states the leakage-safe finite-candidate target-RRI object, the conditional headroom/recovery hypothesis, the negative planning-result rule, and non-claims around continuous control, online RL, real-device deployment, and proxy objectives. Reframed the value-model rationale as "Design Principle: Symmetries of the Decision Problem" and kept geometric deep learning as design vocabulary rather than implementation evidence.

Moved rollout support coverage and evidence gates ahead of the Bellman/Q-loss details, so `Q_H` training is presented after the data/evidence preconditions. Preserved the six RQs, but labeled RQ5/RQ6 as scale and escalation questions. Moved failure interpretation before the Gantt.

## Literature And Citation Notes

Added claim-site citations already present in `docs/references.bib`: ASE/Project Aria, EFM3D/EVL, VIN-NBV, GenNBV, Hestia, PyTorch3D renderer docs, DeepSets, Set Transformer, QCNet, EGNN, e3nn spherical harmonics, SCONE, MACARONS, DQN/Double-DQN, and submodular sensing. No new bibliography key was required in this pass.

The V1 leakage boundary was manually checked against `docs/contents/thesis/questions.qmd` lines 226-244 because the known litkg claim-check path can misclassify the canonical V1 statement. Those lines state that target selection and model inputs use actor-visible predicted/observed descriptors, while GT crops, GT mesh geometry, and GT OBBs are not V1 actor-visible descriptors.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-distillation-lit-review.pdf --root .`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-lit-review-pages --root docs --pages 4-18 --ppi 220`
- Visual inspection of rendered pages covering the thesis contract, RQ figure, symmetry/value-model pages, support/evidence pages, literature ledger, and roadmap.
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ docs/typst/shared`
- `make kg-claim-check KG_CLAIM="VIN-NBV motivates ARIA-NBV's RRI objective and one-step greedy candidate ranking, but does not establish target-conditioned multi-step planning on ASE."` -> supported, confidence 1.0.
- `make kg-claim-check KG_CLAIM="ARIA-NBV Q_H is interpreted as planning only after positive oracle-lookahead headroom and is evaluated by oracle re-scoring under matched budgets."` -> supported, confidence 1.0.
- `git diff --check -- docs/typst/thesis/advisor_distillation.typ`

## Canonical State Impact

No canonical memory update is needed. The revision restates existing thesis direction from roadmap/questions and does not change the target-RRI, V1 leakage, finite-candidate `Q_H`, or bridge/future-work scope.

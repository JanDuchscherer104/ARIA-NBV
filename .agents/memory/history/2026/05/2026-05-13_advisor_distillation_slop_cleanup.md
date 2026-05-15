---
id: 2026-05-13_advisor_distillation_slop_cleanup
date: 2026-05-13
title: "Advisor Distillation Slop Cleanup"
status: done
topics: [thesis, typst, advisor-handout, scientific-writing]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
---

# Advisor Distillation Slop Cleanup

## Task

The advisor handout was reviewed for wording that made it read like AI-generated planning output instead of a scientific memo. The cleanup targeted internal process language such as "research contract", "rescue path", repeated "gate/contract/bridge/core" wording, and caption prose that described project-management rules rather than scientific dependencies.

## Method

The pass kept the thesis spine unchanged: target-specific RRI, actor-visible target descriptors, calibrated myopic scoring, oracle-lookahead headroom, and residual dueling `Q_H` recovery. Edits were limited to the handout prose and captions. The title was changed to "ARIA-NBV: Target-Conditioned Thesis Plan", the opening claim was rewritten as implementation state plus experiment logic, the geometric-ML section was recast as a structured model over candidate sets, poses, directional visibility, and target queries, and the evidence/adoption/roadmap captions were rewritten to state what each table supports.

## Outputs

- Replaced bureaucratic "advisor research contract" phrasing with advisor-memo wording.
- Replaced status narration with factual current-state and planned-extension prose.
- Reduced internal process terms around contracts, gates, bridges, rescue paths, and thesis spine.
- Strengthened the geometric inductive-bias paragraph so the architecture reads as a principled model rather than a module list.
- Preserved the conditional headroom interpretation and made architecture variants diagnostic when headroom is absent.
- Kept the `S^2` directional-memory idea visible as accumulated observability rather than pose encoding.
- Tightened the literature/adoption and risk captions without changing cited roles or thesis scope.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-distillation-slop-clean.pdf --root .`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-slop-clean-pages --root docs --pages 1-18 --ppi 220`
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ`
- Rendered pages 1, 4, 5, 8, 9, 13, and 14 were visually inspected for title, opening claim, formal model, geometric table, equations, evaluation tables, and literature ledger layout.

## Canonical State Impact

No canonical state update is needed. The work is a wording and advisor-presentation cleanup of an already established thesis plan, not a scope, terminology, or implementation change.

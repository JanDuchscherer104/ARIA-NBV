---
id: 2026-05-12_advisor_distillation_todo_resolution
date: 2026-05-12
title: "Advisor Distillation TODO Resolution"
status: done
topics: [typst, thesis, advisor-distillation, notation, kg-claim-check]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
  - docs/typst/shared/equations/entity.typ
---

## Task

Resolved the inline TODOs in `docs/typst/thesis/advisor_distillation.typ` without expanding the handout into the full proposal.

## Method

Reworked the opening handout contract around a finite-horizon masked-candidate NBV MDP, with explicit historic, counterfactual, and oracle state variants. Clarified V0/V1 target protocols, actor-visible target selection, deterministic GT matching, target-cropped point-mesh RRI, candidate selection, candidate mixture families, and the six research-question nodes. Removed the unused displayed target-match acceptance predicate from shared equations after replacing it with prose acceptance criteria in the handout.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-distillation-todos.pdf --root .`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-todos-pages --root docs --pages 1-8 --ppi 220`
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ docs/typst/shared`
- `rg -n "TODO|FIXME|\\?\\?\\?|target_match_acceptance|Q_\\(H,theta\\)|s_obs|~" docs/typst/thesis/advisor_distillation.typ docs/typst/shared/equations/entity.typ`
- `cd docs && typst compile typst/thesis/proposal.typ /tmp/proposal-shared-regression.pdf --root .`
- KG claim checks supported the V1 leakage boundary, stochastic selection versus deterministic GT matching distinction, and headroom-gated `Q_H` evaluation contract.

Strict hygiene still reports advisory-only pre-existing matches in shared generated glossary metadata and shared symbol declarations; no blocking matches were introduced.

## Canonical State Impact

No canonical state update is required. The changes align the advisor handout with existing canonical thesis direction in `docs/contents/thesis/questions.qmd` and `.agents/memory/state/DECISIONS.md`.

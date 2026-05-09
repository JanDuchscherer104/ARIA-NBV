---
id: 2026-05-09_rq_alignment_scaling_escalation
date: 2026-05-09
title: "Research Question Alignment And Scaling Escalation"
status: done
topics: [thesis, questions, roadmap, advisor, typst]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/thesis/questions.qmd
  - docs/contents/thesis/roadmap.qmd
  - docs/typst/thesis/advisor_distillation.typ
---

## Task

Aligned the current research-question page, roadmap, and advisor distillation around the inline review intent from commit `1497283`.

## Output

- Recast `questions.qmd` around six content-level RQs: objective/metrics, target/matching, candidate and rollout support, headroom-gated `Q_H`, scaling, and online/continuous escalation.
- Promoted scaling from a shared protocol detail to a real RQ while preserving mesh/oracle target-RRI supervision as the thesis-grade evidence contract.
- Reframed online discrete `Q_H` and continuous target-then-pose control as lower-priority escalation RQs after finite-candidate evidence, not bridge wording or substitutes for M5.
- Updated the advisor handout with a Typst-native causal RQ dependency DAG and concise six-RQ enumeration.

## Verification

Planned verification includes rendering the changed Quarto pages, compiling and inspecting the advisor handout, running KG claim checks for the changed thesis claims, and scoped whitespace checks.

## Canonical State Impact

No separate canonical memory update was made because `docs/contents/thesis/questions.qmd` and `docs/contents/thesis/roadmap.qmd` are the current thesis-direction owners for this change.

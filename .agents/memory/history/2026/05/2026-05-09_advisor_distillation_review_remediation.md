---
id: 2026-05-09_advisor_distillation_review_remediation
date: 2026-05-09
title: "Advisor Distillation Review Remediation"
status: done
topics: [typst, thesis, advisor, proposal]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
  - .agents/memory/history/2026/05/2026-05-09_advisor_distillation_review_remediation.md
---

## Task

Implemented the GPT-5.5 advisor-distillation review plan for the single-file
Typst advisor handout without changing roadmap, questions, proposal, or
bibliography surfaces.

## Changes

- Reframed the handout opening around the scientific gap, thesis claim, and
  target-RRI oracle / lookahead-headroom / actor-visible-recovery spine.
- Reworded the three RQs so the one-step target scorer is the myopic control
  rather than a standalone central research question.
- Removed the local `display-eq` helper from the handout and converted displayed
  equations to native Typst math blocks with numbering.
- Renamed and subordinated the architecture section as planned value-model
  design; kept R6D pose encoding and separate `S^2` directional memory while
  compressing speculative alternatives into the ablation ladder.
- Updated roadmap/Gantt wording to be advisor-readable and less internal.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-distillation-review-fixed.pdf --root .`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i /home/jd/repos/ARIA-NBV/docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-review-fixed-pages --root /home/jd/repos/ARIA-NBV/docs --pages 4-13 --ppi 220`
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict /home/jd/repos/ARIA-NBV/docs/typst/thesis/advisor_distillation.typ`
- `make kg-claim-check` for GT-crop actor invisibility, lookahead-before-Q_H
  interpretation, oracle re-scoring under equal budgets, and bridge/future
  scope claims.

## Canonical State Impact

No canonical state update is needed. The edit aligns the handout with existing
roadmap/questions/current-memory decisions instead of introducing new thesis
scope.

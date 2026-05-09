---
id: 2026-05-09_questions_drift_remediation
date: 2026-05-09
title: "Questions Drift Remediation"
status: done
topics: [thesis, questions, roadmap, advisor, q-h]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/thesis/questions.qmd
  - docs/contents/thesis/roadmap.qmd
  - docs/typst/thesis/advisor_distillation.typ
---

# Questions Drift Remediation

Applied the GPT review-derived drift fixes without reverting the six-RQ
structure. The current thesis narrative now states that `Q_H` predicts bounded
cumulative target-specific RRI while endpoint target-quality gain remains the
primary fixed-budget evaluation metric.

The pass restored the crop-operator/error contract, descriptor taxonomy,
one-step scorer evidence gate, strict positive-headroom `Q_H` success bar,
setup-specific no-headroom interpretation, two-tier invalidity handling, and
explicit scale/storage reporting rules. The roadmap remains the owner for the
longer operational scale and storage contract.

No canonical memory update is required because the durable decisions were
already present in current thesis docs and `.agents/memory/state/`; this pass
aligned the public surfaces with those decisions.

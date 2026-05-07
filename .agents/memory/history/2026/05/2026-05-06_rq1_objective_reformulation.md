---
id: 2026-05-06_rq1_objective_reformulation
date: 2026-05-06
title: "RQ1 Objective Reformulation"
status: done
topics: [thesis, rri, qh, docs, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/OPEN_QUESTIONS.md
files_touched:
  - docs/contents/thesis/questions.qmd
  - .agents/memory/state/OPEN_QUESTIONS.md
---

## Task

Reformulated RQ1 after local review showed that naming the sum of per-step
state-relative target RRI values as `J_e` made the trajectory objective
horizon-dependent and not interpretable as a bounded endpoint quality score.

## Output

- Replaced the old `J_e = sum RRI_e(...)` expression with endpoint
  target-quality gain `J_{e,Delta}^{(H)}` under a fixed horizon and acquisition
  budget.
- Kept `G_0^{(H)}` as the finite-horizon cumulative target-RRI return for
  rollout ranking and `Q_H` supervision.
- Added log target-error gain `L_e^{(H)}` as an explicit open ablation for scale
  and stage-stability analysis.
- Captured unresolved gamma, epsilon, clipping, log-gain status, and
  near-solved-target eligibility details in canonical open questions.

## Verification

- `make qmd-frontmatter-check`: passed.
- `cd docs && quarto render contents/thesis/questions.qmd`: passed.
- `make kg-claim-check KG_CLAIM="RQ1 separates endpoint target-quality gain J_e from finite-horizon cumulative target-RRI return G_H for Q_H training and treats log target-error gain as an open ablation"`:
  exited 0; output surfaced the expected stale/missing KG backend risk flags but
  returned relevant thesis-question and memory evidence.
- `make check-agent-memory`: passed.

## Canonical State Impact

The durable current-truth change is public in `questions.qmd`; the remaining
advisor-facing choices are tracked in `OPEN_QUESTIONS.md`. No glossary term was
added because existing target-RRI reward and finite-horizon return terms already
cover the training-side notation.

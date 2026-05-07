---
id: 2026-05-06_rl_planning_docs_dedup
date: 2026-05-06
title: "RL Planning Docs De-Duplication"
status: done
topics: [docs, theory, literature, q-learning, rollout]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/theory/rl_planning.qmd
  - docs/contents/literature/rl_planning.qmd
  - docs/contents/literature/index.qmd
  - docs/contents/thesis/questions.qmd
  - docs/contents/thesis/roadmap.qmd
  - docs/typst/shared/glossary.typ
  - docs/contents/glossary.qmd
  - docs/glossary/terms.yml
  - docs/_generated/context/glossary.jsonl
---

## Task

Split duplicate RL planning content into one implementation-facing contract page
and one source-backed literature-distillation page.

## Method

`docs/contents/theory/rl_planning.qmd` now owns the formal finite-candidate
rollout and `Q_H` contract: state variants, candidate action set, transition,
reward, return, baselines, replay row fields, DQN/Double-DQN targets, IQL
ablation shape, and acceptance checks. `docs/contents/literature/rl_planning.qmd`
now owns source-backed thesis-writing distillation for DQN, Double DQN, IQL,
Trajectory Transformer, Gumbel-Top-k, soft Q, PPO, and SAC.

Glossary links were regenerated so formal `Q_H` navigation points to the theory
contract before the literature rationale.

## Verification

Verification run:

- `make glossary`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/theory/rl_planning.qmd`
- `cd docs && quarto render contents/literature/rl_planning.qmd`
- `cd docs && quarto render contents/literature/index.qmd`
- `cd docs && quarto render contents/thesis/questions.qmd`
- `cd docs && quarto render contents/thesis/roadmap.qmd`
- `cd docs && quarto render contents/glossary.qmd`
- `make kg-claim-check KG_CLAIM="ARIA-NBV separates RL literature source distillation from the finite-candidate rollout and Q_H implementation contract"`
- `make check-agent-memory`

## Canonical State Impact

No `.agents/memory/state/` file needed a new durable thesis decision. The public
source-of-truth split is represented directly in the two Quarto pages and the
glossary source.

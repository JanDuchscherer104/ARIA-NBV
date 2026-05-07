---
id: 2026-05-06_thesis_sharpening_revision
date: 2026-05-06
title: "Thesis Sharpening Revision"
status: done
topics: [thesis, proposal, qh, bibliography, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/OPEN_QUESTIONS.md
files_touched:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/OPEN_QUESTIONS.md
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/skills/counterfactual-rollout-planner/SKILL.md
  - docs/_quarto.yml
  - docs/contents/thesis/questions.qmd
  - docs/contents/thesis/roadmap.qmd
  - docs/literature/README.md
  - docs/literature/sources.jsonl
  - docs/references.bib
  - docs/typst/thesis/proposal.typ
  - docs/typst/thesis/sections/proposal/01-motivation.typ
  - docs/typst/thesis/sections/proposal/02-problem.typ
  - docs/typst/thesis/sections/proposal/02-related-work.typ
  - docs/typst/thesis/sections/proposal/03-objectives.typ
  - docs/typst/thesis/sections/proposal/04-method.typ
  - docs/typst/thesis/sections/proposal/05-schedule.typ
  - docs/typst/thesis/sections/proposal/06-outline.typ
---

## Task

Persist Jan's thesis-sharpening decisions and align the advisor-facing thesis
surfaces around target-conditioned, RRI-based multi-step NBV with mandatory
candidate-query Transformer `Q_H`.

## Outputs

- Added the dated thesis-sharpening decision block to `DECISIONS.md`.
- Updated `PROJECT_STATE.md` and `OPEN_QUESTIONS.md` so current truth no
  longer frames `Q_H` as generic fitted Double-Q or optional late RL.
- Revised `questions.qmd` and `roadmap.qmd` around the candidate-query `Q_H`,
  V1 OBS-SEL / PRED-Q / GT-EVAL, geometry-only counterfactual state, required
  rollout sources, and explicit coverage reporting.
- Condensed the Typst proposal to a compact problem/objectives/method contract
  and regenerated the proposal PDF.
- Added CQL, BCQ, Decision Transformer, POMCP, submodularity, and
  submodular-NBV bibliography entries; added only CQL, BCQ, and Decision
  Transformer to `docs/literature/sources.jsonl`.
- Amended existing agents DB records and the rollout-planner skill instead of
  creating duplicate backlog entries.

## Verification

- `make qmd-frontmatter-check` passed.
- The exact multi-file Quarto command failed because this Quarto CLI passed the
  additional QMD paths through to Pandoc for the first document. Rendering
  `questions.qmd`, `roadmap.qmd`, and `m1_contract_report.qmd` individually
  passed.
- `make proposal-pdf` passed after fixing a Typst `#RRI-based` parse issue.
- `make agents-db AGENTS_ARGS='validate'` and `make agents-db` passed.
- `make kg-claim-check KG_CLAIM="ARIA-NBV's thesis core is target-conditioned finite-candidate candidate-query Transformer Q_H over observed/predicted target inputs and geometry-only counterfactual rollouts"` passed and surfaced stale rollout-planner wording, which was then updated.
- `python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" .agents/skills/counterfactual-rollout-planner` passed.
- `make check-agent-memory` passed.

## Canonical State Impact

The durable thesis contract is now persisted in canonical memory, public
Quarto thesis pages, the Typst proposal, bibliography/source manifests, agents
DB, and the relevant rollout-planner skill.

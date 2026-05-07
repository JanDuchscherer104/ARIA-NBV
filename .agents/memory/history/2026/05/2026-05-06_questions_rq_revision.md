---
id: 2026-05-06_questions_rq_revision
date: 2026-05-06
title: "Questions QMD RQ Revision"
status: done
topics: [thesis, questions, roadmap, glossary, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/OPEN_QUESTIONS.md
files_touched:
  - docs/contents/thesis/questions.qmd
  - docs/contents/thesis/roadmap.qmd
  - docs/typst/shared/glossary.typ
  - docs/contents/glossary.qmd
  - docs/glossary/terms.yml
  - docs/_generated/context/glossary.jsonl
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/OPEN_QUESTIONS.md
---

## Task

Implement the five-question thesis RQ revision after grilling the human inline
TODOs in `docs/contents/thesis/questions.qmd`.

## Method

Rewrote the questions page around objective/metrics, actor-visible target
representation, finite candidate/action space, fitted finite-candidate
`Q_H`, and bridge design. Removed standalone RQ4/RQ6/RQ7/RQ8 while preserving
legacy anchors for existing links. Converted invalidity and scale into shared
evidence/protocol constraints. Updated roadmap language and glossary links to
the new anchors.

## Canonical State Impact

`DECISIONS.md` now records that the one-step target-conditioned scorer is a
required baseline rather than a standalone RQ, scaling is an evidence protocol,
online discrete `Q_H` is the first bridge step, and compact actor-visible crop
descriptors are the first target-input ablation. `OPEN_QUESTIONS.md` records the
remaining supervisor-facing choices around crop descriptors, learned invalidity,
Q_H thresholds/normalization, and online discrete `Q_H` scope.

## Verification

Verification commands were run after implementation:

- `make glossary`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/thesis/questions.qmd`
- `cd docs && quarto render contents/thesis/roadmap.qmd`
- `cd docs && quarto render contents/glossary.qmd`
- `make kg-claim-check KG_CLAIM="ARIA-NBV thesis questions separate objective, actor-visible target representation, finite candidates, fitted Q_H planning, and bridge design while treating invalidity and scale as shared protocol constraints"`
- `make check-agent-memory`

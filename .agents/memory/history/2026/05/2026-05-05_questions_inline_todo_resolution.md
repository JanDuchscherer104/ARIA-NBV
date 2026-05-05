---
id: 2026-05-05_questions_inline_todo_resolution
date: 2026-05-05
title: "Questions Inline TODO Resolution"
status: done
topics: [thesis, questions, glossary, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - docs/contents/thesis/questions.qmd
  - docs/typst/shared/glossary.typ
  - .agents/memory/state/DECISIONS.md
---

## Task

Resolved the human inline TODOs in the thesis research-questions page while
preserving the eight-question structure and the hard fitted-Q_H thesis core.

## Method

Updated RQ1 to use a quality-only cumulative target-RRI objective first, split
RQ2 into actor-visible target descriptors and encoding functions, aligned RQ3
with package-supported candidate-generation modes, and made RQ4 a one-step
target-conditioned scorer question only. Added OBS-SEL, PRED-Q, and GT-EVAL to
the canonical glossary source and regenerated derived glossary artifacts.

## Findings

The existing candidate-generation package already supplies the thesis-core
candidate vocabulary for the docs pass: TARGET_POINT, RADIAL_AWAY,
RADIAL_TOWARDS, FORWARD_RIG, UNIFORM_SPHERE, FORWARD_POWERSPHERICAL, and
bounded view jitter. Frontier or missing-surface samplers remain stretch until
implemented and validated.

## Verification

Passed:

- `make glossary`
- `cd docs && quarto render contents/thesis/questions.qmd`
- `cd docs && quarto render contents/glossary.qmd`
- `make check-agent-memory`
- `git diff --check -- docs/contents/thesis/questions.qmd docs/typst/shared/glossary.typ .agents/memory/state/DECISIONS.md`

Additional check:

- `make qmd-frontmatter-check` failed on pre-existing unrelated metadata in
  `docs/contents/ideas.qmd` (`audience: internal`, `status: archive`).

## Canonical State Impact

Updated DECISIONS.md to lock the scorer-only RQ4 split and package-supported
candidate-generation vocabulary.

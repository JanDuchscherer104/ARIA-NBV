---
id: 2026-05-05_thesis_hard_qh_alignment
date: 2026-05-05
title: "Thesis Hard QH Alignment"
status: done
topics: [thesis, roadmap, questions, q-learning, agents-db]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/OPEN_QUESTIONS.md
  - .agents/memory/state/PROJECT_STATE.md
  - docs/contents/thesis/roadmap.qmd
  - docs/contents/thesis/questions.qmd
---

# Thesis Hard QH Alignment

## Task
Implemented the hard thesis-scope alignment plan that makes ARIA-NBV a
target-conditioned, quality-driven ASE/EFM NBV thesis with mandatory fitted
Double-Q / Q_H over finite candidate sets.

## Outputs
- Updated agents DB records so Q_H, V1 observed-target / GT-label protocol,
  Zarr-first rollout/Q storage, LRZ scale gates, target selection, candidate
  mixtures, invalidity masks, and proposal freeze are tracked explicitly.
- Updated thesis roadmap and research questions from optional value/RL gating
  to the stronger sequence: proposal/M1, target contracts, rollout/Q storage,
  full-scale ASE generation, and M5 Q_H success bar.
- Updated durable memory so future agents retrieve the new decisions instead
  of the older optional value/RL framing.

## Verification
Passed after edits:
- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `make check-agent-memory`
- `cd docs && quarto render contents/thesis/roadmap.qmd`
- `cd docs && quarto render contents/thesis/questions.qmd`

## Canonical State Impact
Canonical state was updated directly in `DECISIONS.md`, `PROJECT_STATE.md`, and
`OPEN_QUESTIONS.md`; no additional canonical updates are pending from this
debrief.

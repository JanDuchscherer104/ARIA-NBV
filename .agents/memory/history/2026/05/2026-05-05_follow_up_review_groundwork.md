---
id: 2026-05-05_follow_up_review_groundwork
date: 2026-05-05
title: "Follow-Up Review Groundwork"
status: done
topics: [thesis, roadmap, questions, agents-db, proposal]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/issues.toml
  - .agents/todos.toml
  - docs/contents/thesis/roadmap.qmd
  - docs/contents/thesis/questions.qmd
---

## Task

Converted `.agents/work/alignment-distillation-follow-up-review-gpt55pro.md`
into local planning ground truth without editing `docs/typst/thesis/proposal.typ`
or creating GitHub issues.

## Outputs

- Added `issue-027` for thesis proposal freeze and citation hygiene.
- Added `todo-049`, `todo-050`, and `todo-051` for proposal freeze,
  bibliography audit, and GitHub blocker mirroring.
- Narrowed `issue-012` to the two remaining research-skill gaps.
- Narrowed `issue-018` to large-scale rollout dataset storage/reporting.
- Strengthened `todo-007` with an M1 contract report artifact and Rerun smoke
  proof path.
- Updated the thesis roadmap and research questions with proposal freeze,
  M1 evidence, deterministic rollout ordering, thesis boundary, and supervision
  coverage reporting.

## Scope Notes

The worktree was already dirty. This pass intentionally avoided unrelated Rerun,
config, literature, archive, and litkg submodule changes.

## Verification

- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `make check-agent-memory`
- `cd docs && quarto render contents/thesis/roadmap.qmd`
- `cd docs && quarto render contents/thesis/questions.qmd`

The combined two-file Quarto command form was not accepted by this Quarto
invocation, so the two touched pages were rendered separately.

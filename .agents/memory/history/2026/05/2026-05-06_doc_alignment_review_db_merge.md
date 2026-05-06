---
id: 2026-05-06_doc_alignment_review_db_merge
date: 2026-05-06
title: "Doc Alignment Review DB Merge"
status: done
topics: [agents-db, docs, thesis, review, qmd-frontmatter]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/todos.toml
  - .agents/refactors.toml
  - docs/contents/ideas.qmd
  - scripts/validate_qmd_frontmatter.py
---

## Task

Converted `.agents/work/alignment-distillation/03-doc-alignment-review-gpt55pro.md`
into curated agents DB follow-up work after plan-grilling the review decisions.

## Method

Validated the review against the current branch before creating backlog work:
the API reference links and VIN figure called out by the review already exist
locally, and roadmap/questions render individually. The actual merge therefore
amended existing M1, CI, docs-pruning, bibliography, rollout, storage, scale,
and Q_H todos, and added `todo-055` for one focused source-backed
doc-alignment polish pass.

`docs/contents/ideas.qmd` remains rendered human-owned scratch/archive content.
Its frontmatter now uses `phase: archive`, `audience: public`, `status:
scratch`, `owner: jan`, and `agent_editing: read-only`. The QMD validator was
updated narrowly so rendered archive scratch is accepted only when the page is
marked agent read-only.

## Verification

Passed:

- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/ideas.qmd`
- `cd docs && quarto render contents/thesis/roadmap.qmd`
- `cd docs && quarto render contents/thesis/questions.qmd`
- conflict-marker search over touched docs/backlog surfaces

## Canonical Impact

No canonical memory update is needed. The durable decisions were captured in
the active agents DB records and in the rendered `ideas.qmd` frontmatter
contract.

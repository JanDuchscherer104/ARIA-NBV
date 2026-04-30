---
id: 2026-04-29_prml_crosscheck_agents_db
date: 2026-04-29
title: "PRML Cross-Check Agents DB Update"
status: done
topics: [agents-db, scaffold, backlog, prml-crosscheck]
confidence: high
canonical_updates_needed: []
---

## Task

Parsed the PRML-VSLAM scaffold cross-review against the current thesis roadmap
and recorded missing actionable ARIA-NBV work in the agents DB.

## Method

Started from `docs/contents/roadmap.qmd`, then grounded the update with
`README.md`, `docs/contents/questions.qmd`, `AGENTS.md`, and
`.agents/AGENTS_INTERNAL_DB.md`. Cross-checked
`.agents/work/research-and-cleanup/transcript-02.md` and `transcript-03.md`
against active issues, todos, and refactors.

## Outputs

- Raised CI/pre-commit and setup issues to high priority.
- Added issues for GitHub collaboration scaffold, incomplete ARIA-specific skill
  suite, and missing final experiment evidence registry.
- Added todos for root `make ci`, GitHub templates, geometry/rollout/entity/docs
  and dataset-cache skills, experiment registry, and residual wrong-repo skill
  leakage.
- Added low-priority refactors for optional OMX/Codex templates and KG/memory
  tooling scope.
- Resolved the stale dirty-worktree handoff issue/todo because the cleanup work
  was classified and committed in separate packages.

## Verification

- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`

## Canonical State Impact

No canonical state update was required; this was active backlog maintenance.

---
id: 2026-05-07_rollout_zarr_review_backlog_capture
date: 2026-05-07
title: "Rollout Zarr Review Backlog Capture"
status: done
topics: [agents-db, rollouts, zarr, q-h, review]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/issues.toml
  - .agents/todos.toml
---

## Task

Persisted the critical review findings from the standalone `rollouts.zarr`
tracer bullet into the ARIA-NBV agents DB.

## Method

Amended `issue-018` so the rollout storage debt records the review-discovered
blockers: non-oracle score fallback into Q_H labels, lost multi-target identity,
incorrect root-relative pose math, and insufficient split/lineage persistence.
Added `todo-057` as a concrete high-priority follow-up slice to fix and test
those blockers before `todo-052` can consume rollout stores as trusted Q_H
training data.

## Verification

- `make agents-db`
- `make agents-db AGENTS_ARGS='validate'`
- `make check-agent-memory`

## Canonical State Impact

No durable thesis decision changed. The findings are actionable engineering
debt, so they belong in `.agents/issues.toml` and `.agents/todos.toml` rather
than canonical state memory.

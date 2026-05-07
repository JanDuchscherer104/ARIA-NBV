---
id: 2026-05-07_multimodal_rollout_agents_db_alignment
date: 2026-05-07
title: "Multimodal Rollout Agents DB Alignment"
status: done
topics: [agents-db, rollouts, rerun, streamlit, zarr, q-h]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/issues.toml
  - .agents/todos.toml
---

## Task

Capture the multimodal rollout-inspection alignment as active agents-DB work.
The user requested backlog-only updates, not package implementation.

## Method

Applied the agents-db lane, inspected existing rollout, branching, sharding,
target, Rerun, Streamlit, and Zarr backlog records, then amended existing owners
instead of creating duplicate umbrella issues.

## Outputs

- Amended `issue-018` with the remaining multimodal rollout storage gaps:
  training/audit retention, selected-child RGB-D, one-step modality parity,
  Zarr-tree introspection, and Rerun/Streamlit branch inspection.
- Amended `todo-027`, `todo-033`, and `todo-053` with H3/B3 seeded branching,
  retention-profile-aware sharding, and observed-plus-GT-overlay target
  diagnostics.
- Added `todo-058`, `todo-059`, and `todo-060` for rollout multimodal blocks,
  branch/multimodal viewer inspection, and richer surface-point feature
  ablations.

## Verification

Planned verification:

- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `make check-agent-memory`

## Canonical State Impact

No canonical memory update is required. The durable current truth was already
aligned with V1 strict target visibility, Q_H rollout storage, and Rerun as a
diagnostic surface; this task only added active follow-up work.

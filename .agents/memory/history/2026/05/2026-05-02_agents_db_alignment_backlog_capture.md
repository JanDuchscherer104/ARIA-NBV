---
id: 2026-05-02_agents_db_alignment_backlog_capture
date: 2026-05-02
title: "Agents DB Alignment Backlog Capture"
status: done
topics: [agents-db, rollouts, thesis, litkg, planning]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/issues.toml
  - .agents/todos.toml
---

## Task

Extract issues and action items from the current rollout/RL alignment context
and add them to the ARIA-NBV agents DB.

## Method

Used the `agents-db` skill workflow, read current README, research questions,
root guidance, internal DB, and active backlog, then added consolidated records
instead of duplicating already-covered docs, Rerun, entity-RRI, and M1 contract
items.

## Outputs

- Added issues for rollout trace/storage contracts, stochastic branching,
  target-aware candidate generation and target selection, invalidity semantics,
  cluster-scale generation, litkg/Semantic Scholar drift, and simulator gating.
- Added vertical todo slices for rollout schema, oracle lookahead comparison,
  stochastic selectors, candidate mixtures, target selector, candidate
  diagnostics, invalidity taxonomy, lean storage, deterministic sharding,
  LRZ templates, litkg repair, geometry/order guard tests, VIN evidence-gate
  reporting, and simulator expansion gating.

## Verification

- `make agents-db AGENTS_ARGS='validate'`

## Canonical State Impact

The agents DB now records the actionable work surfaced by the alignment pass.
No additional canonical state update is required.

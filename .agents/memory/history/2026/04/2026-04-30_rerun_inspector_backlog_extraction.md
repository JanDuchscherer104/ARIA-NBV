---
id: 2026-04-30_rerun_inspector_backlog_extraction
date: 2026-04-30
title: "Rerun Inspector Review Action Items Added To Agents DB"
status: done
topics: [agents-db, rerun, offline-store, diagnostics, thesis]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/issues.toml
  - .agents/todos.toml
---

## Task

Converted a code review about offline-store trustworthiness and a TOML-configurable Rerun inspector into local agents DB records.

## Method

Read the required agents-db grounding sources, current package/data/docs guidance, active DB records, and canonical memory state. Reused existing records for the missing `vin_offline` store, M1 contract gate, entity-aware RRI, stale public API debt, and oracle RRI memory refactor instead of duplicating them.

## Outputs

Added active issues for the missing first-class Rerun inspector, frame-safe Rerun geometry logging, and offline-store visual primitive inventory. Updated the missing-store todo to cover Rerun blocking behavior. Added todos for primitive inventory, the inspector CLI/config, frame-safe geometry helpers, Rerun docs/smoke workflow, and non-myopic rollout state/reward/evidence-gate decisions.

## Verification

Ran `make agents-db AGENTS_ARGS='validate'`; validation passed. Ran `make agents-db` to confirm the new records rank correctly.

## Canonical State Impact

No canonical memory update was made. The new backlog item `todo-024` explicitly tracks the later decision/documentation work needed for rollout state, reward, target-aware metric, and VIN evidence gates.

---
id: 2026-05-04_agents_db_alignment_distillation_cleanup
date: 2026-05-04
title: "Agents DB Alignment Distillation Cleanup"
status: done
topics: [agents-db, backlog, rerun, rollout, vin]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/refactors.toml
  - .agents/resolved.toml
---

## Task

Cross-check the alignment distillation against current repo state and update the
agents DB so stale records are resolved and underrepresented action items are
tracked with auditable context.

## Changes

Resolved the now-stale Rerun backlog records for the offline inspector,
visual primitive inventory, and frame-safe Rerun geometry logging. Current code
has `nbv-rerun-inspect`, `OfflineVisualInventory`, Rerun root view-coordinate
logging, zero-candidate and all-invalid safeguards, OBB snippet-to-world
alignment, and camera-context depth logging covered by targeted tests.

Added active backlog coverage for the missing candidate-to-label ordering guard,
the one-scene smoke tutorial, controlled VIN ablation/calibration planning, and
dependency-extra / stale host-path cleanup.

## Verification

- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `make check-agent-memory`

Earlier in the same cross-check, the current Rerun and rollout surfaces were
verified with `cd aria_nbv && uv run pytest tests/rerun_inspector
tests/pose_generation/test_counterfactuals.py`.

## Canonical State Impact

No durable state update is needed. The change is a backlog maintenance update:
current thesis direction and decisions remain unchanged.

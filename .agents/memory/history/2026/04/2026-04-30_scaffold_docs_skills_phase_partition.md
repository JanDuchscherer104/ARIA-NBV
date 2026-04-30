---
id: 2026-04-30_scaffold_docs_skills_phase_partition
date: 2026-04-30
title: "Scaffold Docs Skills Phase Partition"
status: done
topics: [scaffold, skills, docs, agent-memory, quarto]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/DECISIONS.md
---

## Task

Implement the approved scaffold cleanup plan: adopt Matt-style workflows as
ARIA-native skills, keep `.agents/` canonical, keep OMX optional, phase-partition
QMD docs, and retain all kept QMD pages as renderable sources.

## Method

- Added `diagnose-aria` and `plan-grill` as compact workflow skills.
- Folded tracer-bullet TDD, backlog slicing, and zoom-out behavior into
  existing package and agent skills.
- Moved current roadmap/questions into `docs/contents/thesis/` and archived
  ideas/todos/repo-structure pages under `docs/contents/archive/`.
- Moved the Mojo acceleration skill out of active `.agents/skills`.
- Added `.agents/references/human_owner_intent.md` and instruction-capture
  routing.
- Renamed offline-cache slide assets to VIN offline-store naming without
  restoring legacy cache APIs.

## Findings

- `tests/test_panels_dispatcher.py` already imports `offline_dataset` and `rl`;
  the old offline-stats finding is stale.
- `slides_4.typ` already described `VinOfflineDataset`; remaining cleanup was
  asset/data naming around `offline_cache`.
- `.logs` checkpoint/model artifacts are already covered by Git LFS patterns.

## Verification

See the task response for command results.

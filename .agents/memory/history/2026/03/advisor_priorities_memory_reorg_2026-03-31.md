---
id: 2026-03-31_advisor_priorities_memory_reorg
date: 2026-03-31
title: "Advisor Priorities Memory Reorganization"
status: done
topics: [memory, project-state, decisions, open-questions]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/OPEN_QUESTIONS.md
files_touched:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/OPEN_QUESTIONS.md
---

## Task
Reorganize canonical project memory so `PROJECT_STATE.md` becomes a decision-first advisor-facing hub while preserving the existing split between current truth, durable decisions, and unresolved questions.

## Method
Reviewed the current canonical state docs, `docs/contents/ideas.qmd`, and the paper's discussion, extensions, and conclusion sections. Rewrote `PROJECT_STATE.md` around ranked priorities, current blockers, near-term next steps, and deferred extensions; kept `DECISIONS.md` limited to already-adopted repo and project truths; expanded `OPEN_QUESTIONS.md` with advisor-facing scope, planning, data, and modeling questions.

## Findings
The repo already had a good canonical split, but `PROJECT_STATE.md` was too short to serve as an advisor agenda. The most important synthesis outcome is that the strongest current thesis story is geometry-first, RRI-driven non-myopic planning on the current ASE / EFM stack, with hierarchical RL, entity-aware objectives, and semantic-global planning positioned as extensions rather than current claims.

## Verification
Ran `make check-agent-memory`.

## Canonical State Impact
`PROJECT_STATE.md` now acts as the main readable summary of current priorities and recommendations, while `DECISIONS.md` and `OPEN_QUESTIONS.md` retain the detailed stable-versus-unresolved split.

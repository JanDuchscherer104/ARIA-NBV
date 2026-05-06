---
id: 2026-05-06_karpathy_style_agent_scaffold_cleanup
date: 2026-05-06
title: "Karpathy-Style Agent Scaffold Cleanup"
status: done
topics: [scaffold, skills, litkg, agents-db, docs]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - AGENTS.md
  - docs/AGENTS.md
  - aria_nbv/AGENTS.md
  - .agents/skills/
  - .agents/references/
  - .agents/issues.toml
  - .agents/AGENTS_INTERNAL_DB.md
---

## Task

Implement the planned Karpathy-style scaffold cleanup against the review in
`.agents/work/agents-scaffold/review-scaffold-against-kaparthy-skills-gpt55pro.md`.

## Method

Added a compact `agent-behavior` skill, role-split source-order guidance,
skill style and verification references, a litkg quick reference, and routing
metadata on every repo-local skill. Root/docs/package guidance now routes
non-trivial work through `agent-behavior` and separates local discovery,
KG-backed retrieval, and KG implementation.

## Findings

The stale scaffold claims were real: root/docs guidance over-elevated the
seminar paper as all-purpose truth, `AGENTS_INTERNAL_DB.md` still named the NBV
Gym simulator as research core, and rollout skill activation still treated
Gymnasium/SB3 as too central. Those were corrected while keeping Gym/SB3 as a
post-M6 bridge.

## Backlog Impact

Existing issues were amended instead of creating a duplicate scaffold epic.
Existing `todo-056` tracks the deferred litkg-rs implementation needed for the
documented agent-facing context-pack schema.

Follow-up PR review fixes kept the same scope and corrected stale source-order
generation, overly broad docs routing metadata, portable skill validation
wording, Rerun inspector routing globs, and litkg-rs validation commands.

## Verification

Completed:

- validated every `.agents/skills/*` directory with the local skill validator;
- `make check-agent-memory`;
- `make agents-db AGENTS_ARGS='validate'`;
- `make agents-db`;
- `make kg-capabilities KG_FORMAT=json`;
- `make kg-route KG_TASK="review agents-scaffold against Karpathy-style skills and plan scaffold cleanup" KG_FORMAT=json`;
- targeted stale-text search for the old seminar-paper, NBV Gym simulator, and
  Gym/SB3 core-activation wording.
- follow-up validation covered changed skills, agent memory, agents DB, a
  regenerated context index preview, and targeted stale-text searches.

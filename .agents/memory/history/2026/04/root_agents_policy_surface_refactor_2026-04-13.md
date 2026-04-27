---
id: root_agents_policy_surface_refactor_2026-04-13
date: 2026-04-13
title: "Root AGENTS policy-surface refactor"
status: done
topics: [codex, scaffold, memory, agents]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/DECISIONS.md
files_touched:
  - AGENTS.md
  - .agents/references/operator_quick_reference.md
  - .agents/references/agent_memory_templates.md
  - .agents/skills/aria-nbv-context/SKILL.md
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/DECISIONS.md
---

## Task
Refactor the repo-root `AGENTS.md` into a thin repo-wide policy surface while preserving all guidance through explicit owner documents.

## Method
- Mapped each root section to an existing or updated owner surface before removing it from `AGENTS.md`.
- Moved detailed bootstrap and on-demand retrieval guidance into `.agents/skills/aria-nbv-context/SKILL.md`.
- Moved repo-wide execution hygiene into `.agents/references/operator_quick_reference.md`.
- Moved debrief-specific rules into `.agents/references/agent_memory_templates.md`.
- Moved repo-wide tech stack and scaffold-shape truth into canonical state docs.

## Findings
- The prior root file mixed durable policy with detailed workflow, command, and retrieval content that already belonged to nested guides, references, or skills.
- `docs/_generated/context/source_index.md` is not present in this checkout, so the refactored root now treats it as generated hot-path context rather than assuming the file is always present locally.

## Verification
- `make check-agent-memory`
- Manual replacement-owner sanity check for every section removed from `AGENTS.md`

## Canonical State Impact
- Updated `.agents/memory/state/PROJECT_STATE.md` with the repo-wide toolchain section and the thin-root scaffold convention.
- Updated `.agents/memory/state/DECISIONS.md` to record that the repo-root `AGENTS.md` is policy-only and delegates detailed workflows to nested owners.

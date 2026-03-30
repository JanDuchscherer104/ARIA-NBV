---
id: 2026-03-29_scaffold_prompt_memory_capture
date: 2026-03-29
title: "Resolve Scaffold TODOs And Add Prompt-Memory Capture"
status: done
topics: [scaffold, codex, memory, owner-feedback]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/OWNER_DIRECTIVES.md
files_touched:
  - AGENTS.md
  - .agents/memory/README.md
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/OWNER_DIRECTIVES.md
  - .agents/references/agent_memory_templates.md
  - .agents/skills/aria-nbv-context/SKILL.md
  - .agents/skills/aria-nbv-context/references/context_map.md
  - .agents/skills/aria-nbv-context/scripts/nbv_context_index.sh
  - scripts/validate_agent_memory.py
---

# Task

Resolved the remaining root scaffold TODO/FIXME items, added a privileged canonical memory surface for durable owner prompt guidance, and tightened the memory contract so prompt follow-through is enforced for native debriefs.

# Method

- Rewrote the repo-root `AGENTS.md` into a clean bootstrap, commands, invariants, verification, and scope surface with no inline scaffold debris.
- Added `.agents/memory/state/OWNER_DIRECTIVES.md` and promoted durable owner guidance about scaffold shape and memory priority into canonical state.
- Updated memory policy, native debrief templates, and the router skill so `OWNER_DIRECTIVES.md` is part of the default hot path.
- Extended `scripts/validate_agent_memory.py` to fail on unresolved scaffold markers, missing `OWNER_DIRECTIVES.md` references, and native debriefs that omit `## Prompt Follow-Through`.
- Backfilled the existing native March 2026 debriefs with the new prompt-follow-through section so the tighter validator contract applies to the current history set.

# Verification

- `make context`
- `make check-agent-scaffold`
- `make check-agent-memory`
- `aria_nbv/.venv/bin/python scripts/validate_agent_memory.py --self-test`

# Canonical State

- Updated `.agents/memory/state/DECISIONS.md` to record `OWNER_DIRECTIVES.md` as canonical owner-feedback memory and to require prompt follow-through in native debriefs.
- Added `.agents/memory/state/OWNER_DIRECTIVES.md` as the durable store for reusable owner prompt directives.

## Prompt Follow-Through

- Captured the owner requirement that explicit TODOs, issues, core requests, and durable feedback from prompts must be preserved in the memory system.
- Promoted that durable guidance into `.agents/memory/state/OWNER_DIRECTIVES.md` and `.agents/memory/state/DECISIONS.md` instead of leaving it only in this debrief.
- Resolved the root scaffold TODO/FIXME critique by removing inline scaffold debris and replacing it with final command semantics and verification rules in `AGENTS.md`.

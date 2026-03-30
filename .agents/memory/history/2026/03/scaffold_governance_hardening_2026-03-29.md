---
id: 2026-03-29_scaffold_governance_hardening
date: 2026-03-29
title: "Scaffold Governance Hardening"
status: done
topics: [scaffold, governance, docs, validation]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - AGENTS.md
  - aria_nbv/AGENTS.md
  - docs/AGENTS.md
  - .agents/references/tooling_skill_governance.md
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/skills/aria-nbv-context/scripts/nbv_context_index.sh
  - docs/_generated/context/source_index.md
  - scripts/validate_agent_memory.py
  - scripts/validate_mermaid.sh
---

# Task

Hardened the scaffold against runtime tool fetches, added explicit completion criteria to path-local instruction files, and made the governance reference discoverable through the routing surfaces.

# Method

- Replaced the docs Mermaid validation guidance with a repo-owned wrapper command.
- Added a compact root safety rule forbidding runtime network fetches in scaffold guidance.
- Added a governance reference under `.agents/references/` and exposed it through the generated context index.
- Extended scaffold validation to reject remote-fetch patterns and to require `## Verification` plus `## Completion Criteria` in path-local `AGENTS.md` files.

# Verification

- `bash -n scripts/validate_mermaid.sh`
- `aria_nbv/.venv/bin/python -m py_compile scripts/validate_agent_memory.py`
- `aria_nbv/.venv/bin/python scripts/validate_agent_memory.py --self-test`
- `make context`
- `make check-agent-scaffold`
- `make check-agent-memory`

# Canonical State

- Updated `.agents/memory/state/DECISIONS.md` with the local-tooling and required-path-local-sections policy.
- Updated `.agents/memory/state/PROJECT_STATE.md` to record that scaffolded workflows must avoid runtime network fetches.

## Prompt Follow-Through

This note predates the privileged owner-directive memory contract. No additional durable owner prompt items were backfilled here beyond any canonical state updates already recorded in this debrief.

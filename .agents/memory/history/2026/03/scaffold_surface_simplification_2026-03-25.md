---
id: 2026-03-25_scaffold_surface_simplification
date: 2026-03-25
title: "Scaffold Surface Simplification"
status: done
topics: [scaffold, codex, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/GOTCHAS.md
files_touched:
  - AGENTS.md
  - docs/AGENTS.md
  - .agents/skills/aria-nbv-context/SKILL.md
  - .agents/skills/aria-nbv-context/references/context_map.md
  - .agents/skills/aria-nbv-context/scripts/nbv_context_index.sh
  - Makefile
  - scripts/validate_agent_memory.py
---

# Task

Simplified the active Codex scaffold so the hot path uses only routing and canonical-truth surfaces, while heavy generated artifacts remain available as explicit fallback tools.

# What Changed

- Rewrote the repo and docs guidance to make the default bootstrap `main.typ` + canonical state docs + compact `source_index.md`.
- Demoted `context_snapshot.md`, bulk docstrings, UML, and tree artifacts to `make context-heavy` or explicit heavy targets.
- Compressed `source_index.md` generation from a large file inventory into a compact routing index with family counts and reveal commands.
- Trimmed `context_map.md` to only non-obvious cross-surface mappings.
- Added `.agents/references/agent_memory_templates.md`.
- Added `scripts/validate_agent_memory.py` and `make check-agent-memory`.
- Migrated the leftover `.codex/test_failure_investigation_2026-03-24.md` note into structured history.
- Normalized the March 2026 native history notes to use frontmatter plus `canonical_updates_needed`.

# Verification

- `oracle_rri/.venv/bin/python -m py_compile scripts/validate_agent_memory.py`
- `make context`
- `make context-heavy`
- `make check-agent-memory`
- Synthetic failure check:
  - added a temporary `.codex/*.md` note and a temporary native debrief missing `canonical_updates_needed`
  - confirmed `make check-agent-memory` failed on both cases
  - removed the temporary files and confirmed `make check-agent-memory` passed again

# Canonical State

- Updated `.agents/memory/state/DECISIONS.md` to capture the lightweight vs heavy context split and the native debrief contract.
- Updated `.agents/memory/state/PROJECT_STATE.md` to document the new default bootstrap and command-first contract surface.
- Updated `.agents/memory/state/GOTCHAS.md` to distinguish lightweight `make context` refresh from heavy fallback targets.

## Prompt Follow-Through

This note predates the privileged owner-directive memory contract. No additional durable owner prompt items were backfilled here beyond any canonical state updates already recorded in this debrief.

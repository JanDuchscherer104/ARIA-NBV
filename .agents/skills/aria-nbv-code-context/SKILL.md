---
name: aria-nbv-code-context
description: Gather targeted Aria-NBV Python package context. Use when work touches aria_nbv package contracts, configs, CLIs, tests, typed containers, data/RRI/VIN interfaces, or when code symbols/modules need localization before editing.
---

# Aria-NBV Code Context

Use this skill to localize Python code and package contracts. After the module
or symbol set is known, follow the nearest nested `AGENTS.md`.

## Retrieval
1. Read `aria_nbv/AGENTS.md`.
2. Open the nearest module guide when applicable.
3. Use `.agents/references/python_conventions.md` for long-form examples.
4. Use `.agents/memory/state/GOTCHAS.md` for maintained package traps.
5. Start structural discovery with:
   - `make context-contracts`
   - `scripts/nbv_get_context.sh contracts`
   - `scripts/nbv_get_context.sh modules`
   - `scripts/nbv_get_context.sh match <term>`

## Rules
- Keep `PoseTW` and `CameraTW` at semantic boundaries.
- Preserve config-as-factory via `.setup_target()`.
- Treat data-handling, RRI, VIN, Lightning, and rendering payload changes as
  cross-surface contracts.
- Prefer targeted `rg` and file reads once the relevant module set is known.

## Heavy Fallback
Use `make context-heavy`, `make context-uml`, `make context-docstrings`, or
`make context-tree` only after lightweight routing fails to localize the task.

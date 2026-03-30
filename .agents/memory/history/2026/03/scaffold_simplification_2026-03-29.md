---
id: 2026-03-29_scaffold_simplification
date: 2026-03-29
title: "Scaffold Simplification"
status: done
topics: [scaffold, codex, memory, routing]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/GOTCHAS.md
  - .agents/memory/state/OPEN_QUESTIONS.md
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - AGENTS.md
  - Makefile
  - .agents/memory/README.md
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/GOTCHAS.md
  - .agents/memory/state/OPEN_QUESTIONS.md
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/skills/aria-nbv-context/SKILL.md
  - .agents/skills/aria-nbv-context/references/context_map.md
  - .agents/skills/aria-nbv-context/scripts/nbv_context_index.sh
  - .agents/skills/aria-nbv-context/scripts/nbv_literature_index.sh
  - .agents/skills/aria-nbv-context/scripts/nbv_literature_search.sh
  - docs/_generated/context/source_index.md
  - docs/_generated/context/literature_index.md
  - scripts/filter_mermaid.py
  - scripts/format_qmd_lists.py
  - scripts/nbv_typst_includes.py
  - scripts/validate_agent_memory.py
---

# Task

Implemented the staged scaffold simplification plan: repaired stale routing paths, reduced default instruction load, tightened the `aria-nbv-context` skill into a router, and added scaffold validation so drift fails fast.

# Method

- Rewrote the root `AGENTS.md` into a compact bootstrap + commands + invariants + verification + scope surface.
- Removed stale `oracle_rri` references from hot-path canonical state and aligned bootstrap guidance on `PROJECT_STATE.md`.
- Updated the context-generation scripts and routing references to use `docs/literature/` instead of the removed root-level `literature/` path.
- Added `scripts/validate_agent_memory.py` plus `make check-agent-scaffold` so stale paths, missing references, and impossible generated counts are caught automatically.
- Added small script wrappers for existing Makefile targets that referenced missing helper scripts.

# Findings

- The old literature routing bug came from path drift inside the context-generation scripts, not from missing source data.
- `rg --files -g '*.tex' docs/literature` undercounted literature files in the generator path, so the index now uses `find` for stable family and file counts.
- The old `check-agent-memory` target referenced a missing script, so scaffold validation was not actually enforceable until this pass.
- History is now explicitly cold by default; current-state bootstrap is `main.typ` + `PROJECT_STATE.md` + `source_index.md`.

# Verification

- `bash -n .agents/skills/aria-nbv-context/scripts/nbv_context_index.sh .agents/skills/aria-nbv-context/scripts/nbv_literature_index.sh .agents/skills/aria-nbv-context/scripts/nbv_literature_search.sh scripts/nbv_context_index.sh scripts/nbv_literature_index.sh scripts/nbv_literature_search.sh`
- `aria_nbv/.venv/bin/python -m py_compile scripts/validate_agent_memory.py scripts/nbv_typst_includes.py scripts/filter_mermaid.py scripts/format_qmd_lists.py`
- `aria_nbv/.venv/bin/python scripts/validate_agent_memory.py --self-test`
- `make context`
- `make check-agent-scaffold`
- `make check-agent-memory`

# Canonical State

- Updated `.agents/memory/state/PROJECT_STATE.md` to define the reduced bootstrap bundle.
- Updated `.agents/memory/state/DECISIONS.md` to record the compact root scaffold, router-first skill boundary, and scaffold validation target.
- Updated `.agents/memory/state/GOTCHAS.md` to remove stale rename leftovers and keep only project-relevant gotchas.
- Updated `.agents/memory/state/OPEN_QUESTIONS.md` to remove the now-resolved question about the minimum stable Codex bootstrap.

## Prompt Follow-Through

This note predates the privileged owner-directive memory contract. No additional durable owner prompt items were backfilled here beyond any canonical state updates already recorded in this debrief.

---
name: aria-nbv-context
description: Lightweight repo-routing entrypoint for Aria-NBV. Use when a task spans multiple repo surfaces or the right source family is not yet known. Prefer narrower skills once localized: aria-nbv-docs-context for docs/literature, aria-nbv-code-context for package contracts, and aria-nbv-scaffold-maintenance for agent scaffold work.
---

# Aria-NBV Context

Use this as the first routing layer only. Localize the task, then hand off to
the nearest `AGENTS.md` or narrower skill.

## Retrieval Ladder
1. Read `docs/typst/paper/main.typ` for top-level project narrative.
2. Read `.agents/memory/state/PROJECT_STATE.md` for current truth.
3. Use `docs/_generated/context/source_index.md` for the source-family map.
4. Open the nearest nested `AGENTS.md` after the touched subtree is known.
5. Use `.agents/references/` only when conventions, operator aids, or optional
   tool workflows are needed.

## Handoffs
- Docs, Typst, Quarto, bibliography, or literature routing:
  `aria-nbv-docs-context`.
- Python symbols, package contracts, configs, batches, or module ownership:
  `aria-nbv-code-context`.
- Agent guidance, skills, memory, hooks, validation, or `.agents` DB work:
  `aria-nbv-scaffold-maintenance`.

## Commands
- Refresh lightweight context: `make context`
- Inspect source families: `docs/_generated/context/source_index.md`
- Heavy fallback only when needed: `make context-heavy`

## References
- `references/context_map.md` keeps non-obvious concept-to-source routing.

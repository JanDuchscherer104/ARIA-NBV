---
name: aria-nbv-scaffold-maintenance
description: Maintain the Aria-NBV agent scaffold: AGENTS.md files, .agents skills, memory, references, inactive Codex hook templates, agent DB TOML files, scaffold validators, context helpers, and Quarto agent-scaffold publication.
---

# Aria-NBV Scaffold Maintenance

Use this skill for agent guidance, local skills, canonical memory, scaffold
validation, hook templates, and `.agents` DB work.

## Workflow
1. Read root `AGENTS.md`, then the nearest scaffold source being changed.
2. Keep root guidance thin; move subtree-specific details to nested `AGENTS.md`.
3. Keep durable ownership notes in local README/REQUIREMENTS files when that
   prevents `AGENTS.md` bloat.
4. Keep skills concise and self-contained; split workflows instead of growing a
   monolithic skill.
5. Update `scripts/quarto_generate_agent_docs.py` when published scaffold pages
   should include new canonical scaffold sources.
6. Record durable scaffold truth in `.agents/memory/state/` only when current
   truth changes.
7. Record agent/tooling debt in `.agents/issues.toml` and `.agents/todos.toml`;
   move completed records to `.agents/resolved.toml` with `make agents-db`.

## Validation
- Run `make check-agent-scaffold` for scaffold changes.
- Run `make check-agent-memory` for debrief/history-only changes.
- Run `make context` after context-routing changes.
- Run `./scripts/quarto_generate_agent_docs.py` after publication generator
  changes.

## Hooks
- Hook templates are inactive by default and live under
  `.agents/references/codex_hook_templates/`.
- Do not add active `.codex/hooks.json` unless a task explicitly asks to enable
  hooks.

---
id: decisions
updated: 2026-04-15
scope: repo
owner: jan
status: active
tags: [codex, workflow, architecture]
---

# Decisions

## Durable Repo Decisions
- Codex repo guidance uses repo-root and nested `AGENTS.md` files, not `.codex/AGENTS.md`.
- Repo skills live in `.agents/skills/` and use progressive disclosure.
- The old monolithic `aria-nbv-context` skill is now only a lightweight router; narrower repo skills own docs/literature context, code-contract context, and scaffold-maintenance workflows.
- Shared repo guidance must stay machine-portable; operator-specific interpreter or host paths belong in `.agents/references/` or user-local notes, not repo-root or nested `AGENTS.md` files.
- Canonical project memory lives in `.agents/memory/state/`; episodic notes live in `.agents/memory/history/`.
- Agent/tooling debt lives in `.agents/issues.toml`, `.agents/todos.toml`, and `.agents/resolved.toml`; this DB is scoped to agent scaffold, workflow, stale guidance, optional tools, and validation debt rather than the general research backlog.
- Generated context is derived output under `docs/_generated/context/` and should remain untracked.
- `make context` is the lightweight scaffold refresh for `source_index.md`, `literature_index.md`, and `data_contracts.md`.
- `make context-heavy` is explicit fallback for bundled heavy artifacts such as UML, bulk docstrings, and directory trees.
- `docs/_generated/context/source_index.md` is a compact routing index; broad file inventories stay discoverable through commands, not the hot path.
- The Codex hot path stays limited to `docs/typst/paper/main.typ` + `.agents/memory/state/` + `docs/_generated/context/source_index.md`.
- Progressive disclosure routes from the root `AGENTS.md` into package, docs, and module-specific guides only after the touched surface is localized; agents should not load all nested guides up front.
- Root `AGENTS.md` is intentionally thin. Subtree-specific rules belong in nested `AGENTS.md`; durable subsystem ownership belongs in local README or REQUIREMENTS notes when that prevents repeated policy bloat.
- `.agents/references/` holds operator aids and long-form conventions; those docs are on-demand references, not default bootstrap context.
- `docs/typst/paper/main.typ` is the highest-level project truth when it disagrees with Quarto summaries.
- Native debriefs under `.agents/memory/history/` must include `canonical_updates_needed`; existing `status: legacy-imported` notes are grandfathered archive evidence.
- Ad hoc `.codex/*.md` notes are invalid; migrate them into `.agents/memory/history/` or archive them under `archive/codex-legacy/`.
- Verification in shared repo guidance is selected by touched surface rather than by a single global checklist.
- `make check-agent-scaffold` is the broad scaffold health check. `make check-agent-memory` remains the narrower debrief/history hygiene check.
- GitNexus is optional in the Codex scaffold and is documented in `.agents/references/gitnexus_optional.md`; root guidance must not require unavailable GitNexus tooling.
- `.codex/hooks.json` tracks the active startup hook that runs `make context` for new Codex sessions; guardrail hook examples remain inactive templates under `.agents/references/codex_hook_templates/`.

## Technical Decisions
- Runtime objects are instantiated through config `.setup_target()` factories.
- Pose and camera representations use `PoseTW` and `CameraTW`.
- Package verification uses `ruff format`, `ruff check`, and targeted `pytest`.
- The tracked Python workspace and package root are `aria_nbv/` and `aria_nbv/aria_nbv`; repo tooling and docs should refer to that layout.
- Documentation changes should update Quarto/Typst sources directly, not ad hoc notes under `.codex/`.
- The published Quarto site refreshes `aria_nbv` API reference pages from docstrings via `quartodoc` during the Pages workflow, with `docs/reference/index.qmd` as the human-authored landing page.
- The published Quarto site also regenerates maintained agent-scaffold pages from canonical markdown under `AGENTS.md`, nested `AGENTS.md`, `.agents/memory/state/`, `.agents/references/`, `.agents/skills/`, and the active `.agents` DB, while keeping history and archive surfaces out of the site.

---
id: decisions
updated: 2026-03-25
scope: repo
owner: jan
status: active
tags: [codex, workflow, architecture]
---

# Decisions

## Durable Repo Decisions
- Codex repo guidance uses repo-root and nested `AGENTS.md` files, not `.codex/AGENTS.md`.
- Repo skills live in `.agents/skills/` and use progressive disclosure.
- Canonical project memory lives in `.agents/memory/state/`; episodic notes live in `.agents/memory/history/`.
- Generated context is derived output under `docs/_generated/context/` and should remain untracked.
- `make context` is the lightweight scaffold refresh for `source_index.md`, `literature_index.md`, and `data_contracts.md`.
- `make context-heavy` is explicit fallback for bundled heavy artifacts such as UML, bulk docstrings, and directory trees.
- `docs/_generated/context/source_index.md` is a compact routing index; broad file inventories stay discoverable through commands, not the hot path.
- The Codex hot path stays limited to `docs/typst/paper/main.typ` + `.agents/memory/state/` + `docs/_generated/context/source_index.md`.
- `.agents/references/` holds operator aids and long-form conventions; those docs are on-demand references, not default bootstrap context.
- `docs/typst/paper/main.typ` is the highest-level project truth when it disagrees with Quarto summaries.
- Native debriefs under `.agents/memory/history/` must include `canonical_updates_needed`; existing `status: legacy-imported` notes are grandfathered archive evidence.
- Ad hoc `.codex/*.md` notes are invalid; migrate them into `.agents/memory/history/` or archive them under `archive/codex-legacy/`.

## Technical Decisions
- Runtime objects are instantiated through config `.setup_target()` factories.
- Pose and camera representations use `PoseTW` and `CameraTW`.
- Package verification uses `ruff format`, `ruff check`, and targeted `pytest`.
- Documentation changes should update Quarto/Typst sources directly, not ad hoc notes under `.codex/`.

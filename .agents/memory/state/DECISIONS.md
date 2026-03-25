---
id: decisions
updated: 2026-03-24
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
- `make context` remains the standard entrypoint for refreshing repo context, but it now writes modular artifacts instead of a `.codex`-only dump.
- `docs/typst/paper/main.typ` is the highest-level project truth when it disagrees with Quarto summaries.

## Technical Decisions
- Runtime objects are instantiated through config `.setup_target()` factories.
- Pose and camera representations use `PoseTW` and `CameraTW`.
- Package verification uses `ruff format`, `ruff check`, and targeted `pytest`.
- Documentation changes should update Quarto/Typst sources directly, not ad hoc notes under `.codex/`.

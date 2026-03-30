---
id: decisions
updated: 2026-03-29
scope: repo
owner: jan
status: active
tags: [codex, workflow, architecture]
---

# Decisions

## Durable Repo Decisions
- Codex repo guidance uses repo-root and nested `AGENTS.md` files, not `.codex/AGENTS.md`.
- The repo-root `AGENTS.md` stays compact and repo-wide; package-specific implementation guidance belongs in `aria_nbv/AGENTS.md`, docs-specific guidance belongs in `docs/AGENTS.md`, and repeatable retrieval workflows belong in `.agents/skills/`.
- High-risk shared-contract surfaces use deeper boundary guides under `aria_nbv/aria_nbv/{vin,data_handling,rri_metrics}/AGENTS.md` and `docs/typst/paper/AGENTS.md`.
- Repo-managed `AGENTS.md` files carry small YAML frontmatter blocks for scope and summary metadata.
- Canonical project memory lives in `.agents/memory/state/`; episodic notes live in `.agents/memory/history/`.
- `.agents/memory/state/OWNER_DIRECTIVES.md` is the canonical durable store for reusable owner TODOs, issues, core requests, and workflow feedback from prompts.
- The Codex hot path stays limited to `docs/typst/paper/main.typ` + `.agents/memory/state/PROJECT_STATE.md` + `.agents/memory/state/OWNER_DIRECTIVES.md` + `docs/_generated/context/source_index.md`.
- `DECISIONS.md`, `OPEN_QUESTIONS.md`, `GOTCHAS.md`, and `.agents/references/` are on-demand context, not default bootstrap context.
- `.agents/memory/history/` is cold-start excluded by default and may contain stale paths from earlier repo layouts.
- AGENTS and skill workflows must use repo-owned, pinned, or preinstalled local tools; one-shot network package installers and ad-hoc HTTP fetch helpers are not allowed in scaffold guidance.
- Repo-managed scaffold guidance and canonical memory should use repo-relative paths unless an external tool contract requires absolute paths.
- Canonical setup instructions stay in `docs/contents/setup.qmd`; the root scaffold should point there instead of duplicating setup steps.
- `make context` is the lightweight scaffold refresh for `source_index.md`, `literature_index.md`, and `data_contracts.md`.
- `make context-heavy` is explicit fallback for bundled heavy artifacts such as UML, bulk docstrings, and directory trees.
- `make check-agent-scaffold` validates scaffold drift, and `make check-agent-memory` validates scaffold plus memory hygiene.
- `docs/typst/paper/main.typ` is the highest-level project truth when it disagrees with Quarto summaries.
- Native debriefs under `.agents/memory/history/` must include `canonical_updates_needed` and a `## Prompt Follow-Through` section; existing `status: legacy-imported` notes are grandfathered archive evidence.
- Durable owner prompt guidance must be promoted into canonical memory instead of remaining only in chat history or a dated debrief.
- Path-local `AGENTS.md` files under `aria_nbv/` and `docs/` must define both `## Verification` and `## Completion Criteria`.
- Boundary guides under `aria_nbv/aria_nbv/**` and `docs/typst/paper/**` must also define `## Public Contracts` and `## Boundary Rules`.
- Scaffold validation discovers repo-managed `AGENTS.md` files dynamically under `aria_nbv/` and `docs/` rather than relying on a fixed three-file allowlist.
- Generated routing indices should advertise the active path-local boundary guides so localization hands off cleanly to the deepest relevant guide.

## Technical Decisions
- Runtime objects are instantiated through config `.setup_target()` factories.
- Pose and camera representations use `PoseTW` and `CameraTW`.
- Package verification uses `ruff format`, `ruff check`, and targeted `pytest`.
- The tracked Python workspace and package root are `aria_nbv/` and `aria_nbv/aria_nbv`; repo tooling and docs should refer to that layout.
- Documentation changes should update Quarto and Typst sources directly.

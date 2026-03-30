---
scope: repo
applies_to: repository-wide
summary: Repo-wide guidance for agents working across Aria-NBV code, docs, paper, and canonical memory.
---

# NBV Repository Guidance

## Purpose
- Aria-NBV develops a quality-driven next-best-view system for egocentric indoor reconstruction in Aria Synthetic Environments using oracle Relative Reconstruction Improvement (RRI) labels and an EFM3D-backed scorer.
- The current implemented focus is one-step discrete candidate ranking; near-term work prioritizes improving the VIN scorer, scaling oracle supervision, cleaning data handling and docs, and preparing for broader multi-step RL directions.

## Setup & Bootstrap
- Canonical environment and data setup instructions live in `docs/contents/setup.qmd`; point tasks there instead of restating setup steps in the scaffold.
- Start from `docs/typst/paper/main.typ` for the highest-level project narrative.
- Use `.agents/memory/state/PROJECT_STATE.md` as the default project-status read, then `.agents/memory/state/OWNER_DIRECTIVES.md` for durable owner guidance that should shape future work.
- Use `docs/_generated/context/source_index.md` to localize broad tasks; refresh it with `make context` only when it is missing or stale.
- Open `.agents/memory/state/DECISIONS.md`, `.agents/memory/state/OPEN_QUESTIONS.md`, or `.agents/memory/state/GOTCHAS.md` only when the task needs them.
- Use the `aria-nbv-context` skill when the task spans code, docs, paper, or literature and the target files are not yet known; broader routing and reveal workflow live there, while repo-specific command semantics are summarized below.
- Nested `AGENTS.md` files under `aria_nbv/` and `docs/` apply only when working in those trees; once the task is localized, prefer the deepest relevant guide.

## Repo Map
- Repository overview and common CLI entrypoints: `README.md`
- Installable Python package and implementation tree: `aria_nbv/aria_nbv`
- Package tests: `aria_nbv/tests`
- Highest-level research narrative: `docs/typst/paper/main.typ`
- Longer-horizon scratchpad and future directions: `ideas.md`

## Commands
- Lightweight context refresh: `make context` refreshes `source_index.md`, `literature_index.md`, and `data_contracts.md` when routing artifacts are missing or stale.
- Contract surface: `make context-contracts` refreshes the generated contract index for package symbols without running the heavier routing bundle.
- Scaffold validation: `make check-agent-scaffold` validates active scaffold surfaces such as `AGENTS.md`, skills, routing generators, and generated routing indices for drift or unresolved scaffold debris.
- Memory + scaffold validation: `make check-agent-memory` validates canonical memory plus native debrief hygiene, including prompt follow-through and canonical state promotion.

## Core Rules
- Treat `docs/typst/paper/main.typ` as the authoritative high-level description when Quarto docs disagree.
- Use pinned local tools, repo wrappers, or preinstalled commands in `AGENTS.md` and skills; do not embed runtime network fetches.
- Use repo-relative paths in scaffold guidance, canonical memory, and debriefs unless a tool explicitly requires an absolute path.
- Keep documentation aligned with behavior changes.
- Assume the worktree is dirty and work around unrelated changes instead of reverting them.
- Prefer the simplest implementation that cleanly fits the current architecture; remove obsolete interfaces instead of adding compatibility shims unless the task explicitly requires compatibility.
- Treat explicit user TODOs, issues, core requests, and feedback from the prompt as first-class memory inputs.
- Promote durable workflow or process guidance from user prompts into canonical memory, usually `.agents/memory/state/OWNER_DIRECTIVES.md`, instead of leaving it only in chat or a plain debrief.

## Restricted Surfaces
- Canonical memory under `.agents/memory/state/` is durable repo policy; change it only when the task explicitly updates canonical state.
- Scaffold and governance files under `AGENTS.md`, `.agents/references/`, `.agents/skills/`, and `scripts/validate_agent_memory.py` are agent infrastructure; keep edits intentional and validator-backed.
- Generated routing and publish artifacts under `docs/_generated/context/`, `docs/_freeze/`, and `docs/_site/` are derived surfaces; edit the canonical source or generator, then regenerate, unless the task explicitly asks for direct generated-file work.

## Validation Gates
- Run the validation expected by the deepest active path-local `AGENTS.md` for the touched surface.
- Scaffold gate: run `make check-agent-scaffold` after changes to any `AGENTS.md`, `.agents/references/`, `.agents/skills/`, routing generators, generated routing indices, or scaffold validators.
- Memory gate: run `make check-agent-memory` after changes to `.agents/memory/state/`, native debriefs under `.agents/memory/history/`, or scaffold work that also updates canonical memory.
- Routing refresh: run `make context` after changing `nbv_context_index.sh` or when generated routing artifacts need to reflect new scaffold surfaces.

## Maintenance Checklist
- Keep `AGENTS.md` durable, concise, and free of scaffold debris such as inline TODO/FIXME notes.
- Rerun `make check-agent-scaffold` and `make check-agent-memory` after scaffold, canonical memory, or debrief edits.
- Review `.agents/references/tooling_skill_governance.md` before adding new agent-facing tools, wrappers, or skills.
- Promote durable owner feedback into `.agents/memory/state/OWNER_DIRECTIVES.md` instead of leaving it only in chat or history.

## Scope
- Stay within the requested scope; note adjacent issues instead of silently changing them.
- Do not use `git restore` or `git reset --hard` unless the user explicitly asks.
- Do not open `.agents/memory/history/` by default; it is historical evidence, not bootstrap context.
- Do not use `make context-heavy` unless lighter routing failed to localize the answer.

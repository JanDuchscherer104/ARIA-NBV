---
id: decisions
updated: 2026-04-30
scope: repo
owner: jan
status: active
tags: [codex, workflow, architecture]
---

# Decisions

## Durable Repo Decisions
- Codex repo guidance uses repo-root and nested `AGENTS.md` files, not `.codex/AGENTS.md`.
- Repo skills live in `.agents/skills/` and use progressive disclosure.
- Shared repo guidance must stay machine-portable; operator-specific interpreter or host paths belong in `.agents/references/` or user-local notes, not repo-root or nested `AGENTS.md` files.
- Canonical project memory lives in `.agents/memory/state/`; episodic notes live in `.agents/memory/history/`.
- Generated context is derived output under `docs/_generated/context/` and should remain untracked.
- `make context` is the lightweight scaffold refresh for `source_index.md`, `literature_index.md`, and `data_contracts.md`.
- `make context-heavy` is explicit fallback for bundled heavy artifacts such as UML, bulk docstrings, and directory trees.
- `docs/_generated/context/source_index.md` is a compact routing index; broad file inventories stay discoverable through commands, not the hot path.
- The Codex hot path stays limited to `docs/typst/seminar_paper/main.typ` + `.agents/memory/state/` + `docs/_generated/context/source_index.md`.
- Progressive disclosure routes from the root `AGENTS.md` into package, docs, and module-specific guides only after the touched surface is localized; agents should not load all nested guides up front.
- `.agents/references/` holds operator aids and long-form conventions; those docs are on-demand references, not default bootstrap context.
- `docs/typst/seminar_paper/main.typ` is the highest-level project truth when it disagrees with Quarto summaries.
- Native debriefs under `.agents/memory/history/` must include `canonical_updates_needed`; existing `status: legacy-imported` notes are grandfathered archive evidence.
- Ad hoc `.codex/*.md` notes are invalid; migrate them into `.agents/memory/history/` or archive them under `archive/codex-legacy/`.
- Verification in shared repo guidance is selected by touched surface rather than by a single global checklist.

## Technical Decisions
- Runtime objects are instantiated through config `.setup_target()` factories.
- Pose and camera representations use `PoseTW` and `CameraTW`.
- Package verification uses `ruff format`, `ruff check`, and targeted `pytest`.
- The tracked Python workspace and package root are `aria_nbv/` and `aria_nbv/aria_nbv`; repo tooling and docs should refer to that layout.
- Documentation changes should update Quarto/Typst sources directly, not ad hoc notes under `.codex/`.
- The published Quarto site refreshes `aria_nbv` API reference pages from docstrings via `quartodoc` during the Pages workflow, with `docs/reference/index.qmd` as the human-authored landing page.
- Generated agent-scaffold pages are internal operator artifacts under
  `.agents/generated/agent_scaffold/`; the published Quarto site must not
  regenerate or render them.
- Retained QMD docs remain renderable, but current thesis pages use
  `docs/contents/thesis/`, past seminar material uses `docs/contents/seminar/`,
  and only curated public archive summaries use `docs/contents/archive/` with
  explicit `phase`, `audience`, `status`, and `owner` frontmatter. Raw
  scratch/history belongs under `.agents/archive/docs/`.
- Human-owner preferences that are durable but not public narrative or workflow
  rules live in `.agents/references/human_owner_intent.md`.
- OMX remains optional operator orchestration; it does not own canonical memory,
  backlog, docs, or repo state.

## Working Project Decisions
- RRI is the primary project objective for next-best-view research in this repo. Coverage-style objectives remain baselines or diagnostics, not the main thesis target.
- The canonical training and evaluation surface remains discrete candidate ranking anchored on prerecorded ASE trajectory snippets with oracle supervision derived from GT meshes where available.
- Non-myopic work is being introduced incrementally through multi-step counterfactual rollouts, structured evaluator metrics, cumulative-RRI accounting, and a discrete-shell RL scaffold before any continuous-control claim.
- Counterfactual reasoning is currently geometry-first: logged ego-trajectory modalities are the trustworthy historical state, while counterfactual views default to mesh-, depth-, and visibility-derived quantities plus accumulated selected observations until richer synthesis exists.
- Entity-aware objectives, hierarchical control, and semantic-global planning are treated as extensions on top of the current scene-level RRI baseline rather than replacements for it.

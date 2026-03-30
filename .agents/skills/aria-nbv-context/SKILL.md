---
name: aria-nbv-context
description: Route broad Aria-NBV tasks to the smallest relevant set of files across the paper, canonical state, docs, literature, and code. Use when the target files are not yet known. Do not trigger for already-localized one-file edits.
---

# Aria NBV Context

Use this skill as a routing layer, not as a broad reading checklist. Localize the task, then hand off to the narrower workflow or path-local `AGENTS.md`.

## Use When
- The task spans code, docs, paper, literature, or canonical repo state.
- The user needs architectural, methodological, or citation context before editing.
- The target file, module, or doc page is not yet known.

## Do Not Use When
- The user already named the exact file or symbol to edit or review.
- The task is confined to one module under `aria_nbv/`.
- The task is a localized Typst or Quarto edit.
- The task is historical and explicitly asks for prior debriefs; go to `.agents/memory/history/` directly only after checking current state docs.

## Default Retrieval Ladder
1. Open `docs/typst/paper/main.typ`.
2. Open `.agents/memory/state/PROJECT_STATE.md`.
3. Open `.agents/memory/state/OWNER_DIRECTIVES.md`.
4. Open `docs/_generated/context/source_index.md`.
5. Refresh lightweight routing with `make context` only if `source_index.md`, `literature_index.md`, or `data_contracts.md` are missing or stale.
6. Open `.agents/memory/state/DECISIONS.md`, `OPEN_QUESTIONS.md`, or `GOTCHAS.md` only if the task needs them.
7. Open `.agents/references/` only when conventions, operator aids, or templates are needed.
8. Use the lightest source-specific reveal command, then switch to targeted `rg` and exact file reads.

## Reveal Commands
- Code: `scripts/nbv_get_context.sh contracts`, `modules`, or `match <term>`
- Quarto: `scripts/nbv_qmd_outline.sh --compact`
- Typst paper: `scripts/nbv_typst_includes.py --paper --mode outline`
- Literature: `scripts/nbv_literature_index.sh`, then `scripts/nbv_literature_search.sh "<term>"`
- Non-obvious cross-surface routing: `references/context_map.md`

## Routing Rules
- For `aria_nbv/**`, open `aria_nbv/AGENTS.md` once the module set is known.
- For `aria_nbv/aria_nbv/vin/**`, open `aria_nbv/aria_nbv/vin/AGENTS.md`.
- For `aria_nbv/aria_nbv/data_handling/**`, open `aria_nbv/aria_nbv/data_handling/AGENTS.md`.
- For `aria_nbv/aria_nbv/rri_metrics/**`, open `aria_nbv/aria_nbv/rri_metrics/AGENTS.md`.
- For `docs/**`, open `docs/AGENTS.md` once the page or Typst section is known.
- For `docs/typst/paper/**`, open `docs/typst/paper/AGENTS.md`.
- For literature, prefer `docs/literature/tex-src/**` source files over PDFs.
- Open `docs/index.qmd`, `docs/contents/todos.qmd`, `roadmap.qmd`, or `questions.qmd` only for project narrative, roadmap, or active work items.
- Do not use `.agents/memory/history/` for default bootstrap; it is historical evidence and may contain stale paths.
- Do not escalate to `make context-heavy` unless the lighter routing steps failed to localize the answer.

## Handoff
Stop using this skill once the task is localized.

- Handoff to the relevant path-local `AGENTS.md` for code or docs work.
- Handoff to the deepest relevant boundary guide once the task localizes to `vin`, `data_handling`, `rri_metrics`, or `docs/typst/paper`.
- Handoff to `typst-authoring` for Typst implementation.
- Handoff to `scientific-writing` for literature synthesis or citation-heavy prose.

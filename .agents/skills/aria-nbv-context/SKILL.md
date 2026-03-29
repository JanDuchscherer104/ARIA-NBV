---
name: aria-nbv-context
description: Gather targeted context for Aria-NBV from the paper, agent memory, Quarto docs, literature, and source code. Use when the task spans multiple docs or code areas, needs citations or architecture context, or requires locating specific technical details across the repo. Do not trigger for already-localized one-file edits.
---

# Aria NBV Context

## Overview
Use this skill as the repo's discovery-and-routing layer. It should localize the task to the smallest relevant set of files, then hand off to the narrower workflow or skill that will do the actual work.

Progressive disclosure order:
1. Fixed ground truth: `docs/typst/paper/main.typ`
2. Canonical current truth: `.agents/memory/state/`
3. Compact generated routing index: `docs/_generated/context/source_index.md`
4. Agent references in `.agents/references/` when conventions, operator aids, or external dependency lookup are needed
5. Checked-in routing map for non-obvious cross-surface topics
6. Refresh lightweight artifacts with `make context` when `source_index.md`, `literature_index.md`, or `data_contracts.md` are stale or missing
7. Source-specific outline or AST summary tools
8. Raw file reads and targeted `rg`
9. `make context-heavy` or specific heavy targets only when UML, class docstrings, or tree structure are actually needed

## When To Use
- The question spans paper, docs, literature, code, or canonical project state.
- The task needs architectural, methodological, or citation context before editing.
- The target file or symbol is not yet known.

## When Not To Use
- The user already named the exact file to edit or review.
- The task is a localized code change inside one module.
- The task is pure formatting or renaming with no cross-doc or cross-code context need.
- The task is already localized to Typst editing, scientific writing, or a single code module.

## Retrieval Ladder (default)
1. Open `docs/typst/paper/main.typ` first; treat it as the highest-level project ground truth.
2. Read the canonical state docs in `.agents/memory/state/`:
   - `PROJECT_STATE.md`
   - `DECISIONS.md`
   - `OPEN_QUESTIONS.md`
   - `GOTCHAS.md`
3. Open `docs/_generated/context/source_index.md` for the compact source-family map, curated documentation families, and reveal commands.
4. Open agent-facing references in `.agents/references/` when the task needs conventions, operator guidance, or external dependency lookup:
   - `operator_quick_reference.md` for environment recovery, repo hygiene, and ASE/EFM quick references
   - `python_conventions.md`
   - `agent_memory_templates.md` for native debrief work
   - `context7_library_ids.md` only when external library docs are part of the task
5. Open `references/context_map.md` only when the task needs non-obvious cross-surface routing.
6. Run `make context` when `source_index.md`, `literature_index.md`, or `data_contracts.md` are missing or stale.
7. Pick the source family and use the lightest source-specific tool first:
   - Quarto: `scripts/nbv_qmd_outline.sh --compact`
   - Typst paper: `scripts/nbv_typst_includes.py --paper --mode outline`
   - Literature: `scripts/nbv_literature_index.sh`
   - Code: `aria_nbv/AGENTS.md` -> `.agents/references/python_conventions.md` -> `.agents/memory/state/GOTCHAS.md` -> `scripts/nbv_get_context.sh contracts` or `make context-contracts` -> `scripts/nbv_get_context.sh modules` or `scripts/nbv_get_context.sh match <term>`
8. Use targeted `rg` inside the narrowed file set.
9. Run `make context-heavy` or the specific `context-uml`, `context-docstrings`, or `context-tree` targets only when the answer still requires heavyweight generated artifacts.

## Do Not Escalate Early
- Do not run `make context-heavy` when the state docs, references, routing map, source index, outlines, or AST summaries already localize the answer.
- Do not open entire paper sections, Quarto chapters, or literature trees until the outline or index step identifies the exact file.
- Do not open `docs/index.qmd` or `docs/contents/todos.qmd` until the task specifically needs project narrative, priorities, or open work items.
- Do not search `.agents/memory/history/` unless the question is historical, comparative, or explicitly asks for past debriefs.

## Canonical State (`.agents/memory/`)
Treat `.agents/memory/state/` as current truth and `.agents/memory/history/` as optional historical evidence.

Use cases:
- `PROJECT_STATE.md`: current goals, architecture summary, stable conventions
- `DECISIONS.md`: major design choices and resolved tradeoffs
- `OPEN_QUESTIONS.md`: active uncertainties, pending experiments, unresolved design questions
- `GOTCHAS.md`: maintained failure modes, environment pitfalls, and recurring verification traps
- `.agents/memory/history/`: prior debriefs only when the task needs historical comparison or evidence

## Routing Assets
### 1) Checked-in routing map
`references/context_map.md` maps major project concepts to exact entry files across:
- `.agents/memory/`
- `.agents/references/`
- Typst paper
- Quarto docs
- Literature sources
- `aria_nbv` code

Use this before broad text search when the task topic is known but the files are not.

### 2) Generated source index
`scripts/nbv_context_index.sh` writes `docs/_generated/context/source_index.md` with:
- fixed entrypoints and canonical state paths
- a compact source-family map with counts
- curated documentation families and secondary references
- preferred reveal commands
- recommended search commands

Use the source index when the topic is still broad and you need to identify the right source family first.

## Source-Specific Reveal Stages
### Quarto docs (`docs/**/*.qmd`)
- Start with `scripts/nbv_qmd_outline.sh --compact`.
- Use full outline mode only when you need nested section structure.
- After narrowing, open the exact `.qmd` page and use `rg` within that page.

### Typst paper (`docs/typst/paper/**/*.typ`)
- Start with `scripts/nbv_typst_includes.py --paper --mode outline`.
- Use `--mode includes` when you only need the include graph.
- Use `--with-slides` only when the task explicitly touches slides.
- Treat `docs/typst/paper/main.typ` as the source of truth over Quarto summaries.

### Literature (`literature/**`)
- Start with `scripts/nbv_literature_index.sh` to identify the right paper family.
- Use `scripts/nbv_literature_search.sh "<term>"` only after the paper family is known.
- Use `.agents/references/context7_library_ids.md` when external library docs are part of the question.
- Prefer `.tex` and `.bib` source reads over opening large PDFs.

### Source code (`aria_nbv/**`)
- Start with `aria_nbv/AGENTS.md` for binding local package rules.
- Open `.agents/references/python_conventions.md` for long-form typing, docstring, and config examples.
- Open `.agents/memory/state/GOTCHAS.md` for maintained failure modes.
- Start structural discovery with `scripts/nbv_get_context.sh contracts` or `make context-contracts` for data contracts and config contracts.
- Use `scripts/nbv_get_context.sh modules` for a module-level map.
- Use `scripts/nbv_get_context.sh match <term>` to narrow by symbol, module, field, or constant name.
- Use `scripts/nbv_get_context.sh functions` or `classes` only after the module set is narrowed.
- Use raw `rg` and file reads once the relevant module or symbol is known.

## Heavyweight Fallback
`make context-heavy` writes the bundled fallback artifacts under `docs/_generated/context/`:
- `context_snapshot.md`
- `aria_nbv_uml.mmd`
- `aria_nbv_filtered_uml.mmd`
- `aria_nbv_class_docstrings.md`
- `aria_nbv_tree.md`

Prefer the specific `make context-uml`, `make context-docstrings`, or `make context-tree` targets when you need only one heavy artifact. Use these only when lighter retrieval failed to localize the answer.

## Handoffs
This skill should stop being the active surface once the task is localized.

- Handoff to `typst-authoring` when the task becomes a Typst edit or paper/slides implementation.
- Handoff to `scientific-writing` when the task becomes literature synthesis, section drafting, or citation-heavy prose.
- Handoff to the direct code workflow once the target module or symbol set in `aria_nbv` is known.

## Bundled Scripts
- Scripts are self-rooting and can be run from any working directory.
- Convenience wrappers live in `scripts/` and delegate to the skill scripts.
- `scripts/nbv_context_index.sh`
  - writes `docs/_generated/context/source_index.md`
- `make context`
  - refreshes `source_index.md`, `literature_index.md`, and `data_contracts.md`
- `scripts/nbv_qmd_outline.sh [--compact]`
  - outlines Quarto pages; compact mode lists the first heading per page
- `scripts/nbv_typst_includes.py --paper --mode outline|includes`
  - outlines Typst paper includes with repo-relative paths; add `--with-slides` only when needed
- `scripts/nbv_literature_index.sh`
  - writes `docs/_generated/context/literature_index.md`
- `scripts/nbv_literature_search.sh "<term>"`
  - searches literature `.tex`, `.bib`, and `.sty` sources
- `scripts/nbv_get_context.sh modules|packages|classes|functions|contracts|match <term>`
  - AST-based code summaries for `aria_nbv`

## References
- `references/context_map.md` provides concept-to-source routing.

---
scope: docs
applies_to: docs/**
summary: Documentation, Typst, Quarto, and publishing guidance for work under docs/.
---

# Docs Guidance

Apply this file when working under `docs/`.

## Priorities
- Keep Quarto docs aligned to canonical project sources instead of introducing competing top-level narratives.
- Keep `docs/references.bib` as the single bibliography source of truth.
- Preserve established Quarto and Typst structure unless the task explicitly changes it.
- Prefer links to canonical state docs in `.agents/memory/state/` over re-explaining the same guidance in multiple places.
- Open `docs/typst/paper/AGENTS.md` when the task localizes to `docs/typst/paper/**` or when Quarto/slides must align to the paper narrative.

## Default Workflow
- Localize the exact Quarto page or Typst section before opening broad doc trees.
- Use `scripts/nbv_qmd_outline.sh --compact` to localize the exact Quarto page before opening it.
- Use `scripts/nbv_typst_includes.py --paper --mode outline` to localize the exact paper Typst section before opening it.
- Open `docs/index.qmd`, `docs/contents/todos.qmd`, `docs/contents/roadmap.qmd`, and `docs/contents/questions.qmd` only when the task is about project narrative, priorities, or roadmap.
- If you need current project truth beyond the localized docs surface, open the relevant doc in `.agents/memory/state/` instead of scanning broad doc trees.

## Commands
- Context refresh: `make context`
- Quarto render: `cd docs && quarto render .`
- Quarto preview: `cd docs && quarto preview`
- Quarto check: `quarto check`
- Typst paper: `cd docs && typst compile typst/paper/main.typ --root .`
- Typst slides: `cd docs && typst compile typst/slides/<file>.typ --root .`
- Typst fallback on sandboxed snap installs: `/snap/typst/current/bin/typst compile <file>.typ --root docs`
- Mermaid validation: `scripts/validate_mermaid.sh <input.mmd> <output.svg>`
- QMD tree: `make context-qmd-tree`
- Outline-first routing: `scripts/nbv_qmd_outline.sh`, `scripts/nbv_typst_includes.py`

## Diagram Rules
- Validate Mermaid before committing diagram edits.
- For non-trivial Mermaid edits, validate standalone first with `scripts/validate_mermaid.sh /tmp/diagram.mmd /tmp/diagram.svg`.
- Use `{mermaid}` fences in Quarto.
- Use `<br/>` for Mermaid line breaks and Mermaid-safe node ids.

## Writing Rules
- Keep cross-references and bibliography entries synchronized.
- Add new references to `docs/references.bib` when introducing important concepts or papers.
- Replace temporary citation placeholders such as `cite…` before finishing.
- Use links to relevant internal docs or authoritative external references when introducing non-obvious concepts.
- Keep Quarto source files (`*.qmd`) separate from rendered site output. Published HTML belongs under `docs/_site/`, not next to the sources.
- Treat `docs/_freeze/` as tracked execution state for code-backed pages when needed; treat `docs/_site/`, `site_libs/`, `index_files/`, and `*_files/` as generated publish artifacts.
- Do not store generated context or rendered artifacts in tracked docs paths unless the task explicitly requires it.
- For larger doc changes, run the relevant render/compile commands plus `quarto check` before finishing.

## Verification
- For Quarto page changes, run `quarto check`; run `cd docs && quarto render .` when the change affects rendered output or cross-page wiring.
- For Typst changes, run the relevant `typst compile` command for the affected paper or slides surface.
- For Mermaid edits, run `scripts/validate_mermaid.sh` on a standalone `.mmd` file before relying on the diagram in Quarto.
- Confirm citations, cross-references, and internal links remain consistent after substantive doc edits.

## Completion Criteria
- Relevant render, compile, or validation commands were run for the changed docs surface.
- Citations and cross-references are synchronized and no placeholder citations remain.
- No generated publish artifacts were accidentally introduced into tracked docs paths.
- Added or changed diagrams validate with the repo-owned Mermaid wrapper before finish.

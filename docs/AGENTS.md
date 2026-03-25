# Docs Guidance

Apply this file when working under `docs/`.

## Priorities
- Treat `docs/typst/paper/main.typ` as the highest-level project ground truth.
- Keep Quarto docs aligned to the paper instead of introducing competing top-level narratives.
- Keep `docs/references.bib` as the single bibliography source of truth.
- Preserve established Quarto and Typst structure unless the task explicitly changes it.
- Prefer links to canonical state docs in `.agents/memory/state/` over re-explaining the same guidance in multiple places.

## Default Workflow
- Start from `docs/typst/paper/main.typ`.
- Use `scripts/nbv_qmd_outline.sh --compact` to localize the exact Quarto page before opening it.
- Use `scripts/nbv_typst_includes.py --paper --mode outline` to localize the exact Typst section before opening it.
- Open `docs/index.qmd`, `docs/contents/todos.qmd`, `docs/contents/roadmap.qmd`, and `docs/contents/questions.qmd` only when the task is about project narrative, priorities, or roadmap.
- If you need current project truth beyond the paper, open the relevant doc in `.agents/memory/state/` instead of scanning broad doc trees.

## Commands
- Context refresh: `make context`
- Quarto render: `quarto render <file>.qmd --to html`
- Quarto check: `quarto check`
- Typst paper: `cd docs && typst compile typst/paper/main.typ --root .`
- Typst slides: `cd docs && typst compile typst/slides/<file>.typ --root .`
- Typst fallback on sandboxed snap installs: `/snap/typst/current/bin/typst compile <file>.typ --root docs`
- QMD tree: `make context-qmd-tree`
- Outline-first routing: `scripts/nbv_qmd_outline.sh`, `scripts/nbv_typst_includes.py`

## Diagram Rules
- Validate Mermaid before committing diagram edits.
- For non-trivial Mermaid edits, validate standalone first with `npx -y @mermaid-js/mermaid-cli -i /tmp/diagram.mmd -o /tmp/diagram.svg`.
- Use `{mermaid}` fences in Quarto.
- Use `<br/>` for Mermaid line breaks and Mermaid-safe node ids.

## Writing Rules
- Keep cross-references and bibliography entries synchronized.
- Add new references to `docs/references.bib` when introducing important concepts or papers.
- Replace temporary citation placeholders such as `cite…` before finishing.
- Use links to relevant internal docs or authoritative external references when introducing non-obvious concepts.
- Do not store generated context or rendered artifacts in tracked docs paths unless the task explicitly requires it.
- For larger doc changes, run `quarto render` and `quarto check` before finishing.

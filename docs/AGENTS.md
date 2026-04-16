# Docs Guidance

This file applies to work under `docs/` and adds documentation-specific deltas
on top of the root [AGENTS.md](../AGENTS.md). For paper work, also follow
[typst/paper/AGENTS.md](typst/paper/AGENTS.md).

## Sources Of Truth
- `docs/typst/paper/main.typ`: highest-level research narrative.
- `docs/references.bib`: single bibliography source of truth.
- `.agents/memory/state/`: current project truth when the paper needs
  implementation or roadmap context.
- `.agents/references/context7_library_ids.md`: approved external-doc lookup
  identifiers.

## Default Workflow
- Start from `docs/typst/paper/main.typ` for research narrative changes.
- Use `scripts/nbv_qmd_outline.sh --compact` to localize Quarto pages.
- Use `scripts/nbv_typst_includes.py --paper --mode outline` to localize Typst
  paper sections.
- Open `docs/index.qmd`, `docs/contents/todos.qmd`,
  `docs/contents/roadmap.qmd`, and `docs/contents/questions.qmd` only when the
  task is about project narrative, priorities, or roadmap.
- Preserve established Quarto and Typst structure unless the task explicitly
  changes it.

## Rules
- Keep Quarto docs aligned to the paper instead of introducing competing
  top-level narratives.
- Keep cross-references and bibliography entries synchronized.
- Add new references to `docs/references.bib` when introducing important
  concepts or papers.
- Replace temporary citation placeholders before finishing.
- Keep Quarto source files separate from rendered site output. Published HTML
  belongs under `docs/_site/`.
- Treat `docs/contents/resources/agent_scaffold/` as generated-from-source
  content. Refresh it with `./scripts/quarto_generate_agent_docs.py` rather than
  editing generated pages by hand.

## Commands
- Agent scaffold pages: `./scripts/quarto_generate_agent_docs.py`
- API reference pages: `./scripts/quarto_generate_api_docs.sh`
- Quarto render: `cd docs && quarto render .`
- Quarto preview: `cd docs && quarto preview`
- Quarto check: `quarto check`
- Typst paper: `cd docs && typst compile typst/paper/main.typ --root .`
- Typst slides: `cd docs && typst compile typst/slides/<file>.typ --root .`

## Diagrams
- Validate Mermaid before committing diagram edits.
- For non-trivial Mermaid edits, validate standalone with
  `npx -y @mermaid-js/mermaid-cli -i /tmp/diagram.mmd -o /tmp/diagram.svg`.
- Use `{mermaid}` fences in Quarto.
- Use `<br/>` for Mermaid line breaks and Mermaid-safe node ids.

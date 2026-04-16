---
name: aria-nbv-docs-context
description: Gather targeted Aria-NBV documentation, Typst, Quarto, bibliography, and literature context. Use when work touches docs/, docs/typst/, docs/references.bib, literature context, paper/slides narrative, citations, or advisor-facing project prose.
---

# Aria-NBV Docs Context

Use this skill for documentation and research-writing discovery. Stop once the
exact page, section, citation, or figure surface is localized.

## Retrieval
- Paper first: `docs/typst/paper/main.typ`
- Paper outline: `scripts/nbv_typst_includes.py --paper --mode outline`
- Quarto outline: `scripts/nbv_qmd_outline.sh --compact`
- Literature index: `scripts/nbv_literature_index.sh`
- Literature search after narrowing: `scripts/nbv_literature_search.sh "<term>"`

## Rules
- Treat the Typst paper as authoritative over Quarto summaries.
- Use `docs/AGENTS.md` for docs-wide rules and `docs/typst/paper/AGENTS.md`
  for paper-specific deltas.
- Prefer `.tex` and `.bib` source reads over large PDFs.
- Keep generated pages under `docs/contents/resources/agent_scaffold/` derived
  from source markdown via `./scripts/quarto_generate_agent_docs.py`.

## Handoffs
- Use `typst-authoring` once the task becomes a Typst edit.
- Use `scientific-writing` once the task becomes citation-heavy prose drafting.

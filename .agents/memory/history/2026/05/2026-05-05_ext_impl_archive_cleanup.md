---
id: 2026-05-05_ext_impl_archive_cleanup
date: 2026-05-05
title: "Ext-Impl Archive Cleanup"
status: done
topics: [docs, archive, agents-db, glossary, external-stack]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/archive/docs/ext-impl/atek_implementation.qmd
  - .agents/archive/docs/ext-impl/efm3d_implementation.qmd
  - .agents/archive/docs/ext-impl/efm3d_symbol_index.qmd
  - .agents/archive/docs/ext-impl/prj_aria_tools_impl.qmd
  - .agents/references/external_stack_contracts.md
  - .agents/todos.toml
  - .agents/refactors.toml
  - docs/_quarto.yml
  - docs/index.qmd
  - docs/contents/impl/overview.qmd
  - docs/contents/impl/aria_nbv_overview.qmd
  - docs/contents/theory/surface_metrics.qmd
  - docs/typst/shared/glossary.typ
  - docs/contents/glossary.qmd
  - docs/glossary/terms.yml
  - docs/_generated/context/glossary.jsonl
---

## Task

Archived noisy public `docs/contents/ext-impl/*.qmd` pages into `.agents/archive/docs/ext-impl/`, removed their public navigation, and retained only the durable ATEK/EFM3D/Project Aria facts needed for future package and docstring work.

## Method

Moved the raw QMD pages to the internal archive, marked each page deprecated, and added `.agents/references/external_stack_contracts.md` as the compact developer reference. Public implementation navigation now groups current pages by thesis evidence surface: system/package, data-oracle-diagnostics, and scorer/training.

The glossary source was updated to remove public links to archived ext-impl pages and regenerated with `make glossary`. The agents DB now records the cleanup boundary in `todo-001` and `refactor-004`, plus `todo-054` for targeted package docstring enrichment from the distilled external-stack contracts.

## Verification

Passed:

- `make glossary`
- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `cd docs && quarto render index.qmd`
- `cd docs && quarto render contents/glossary.qmd`
- `cd docs && quarto render contents/impl/overview.qmd`
- `cd docs && quarto render contents/theory/surface_metrics.qmd`
- `rg 'contents/ext-impl|External Implementations|ProjectAria Tools Reference|EFM3D Symbol Index|ATEK Implementation Index|EFM3D Implementation Index' docs README.md -g '!_site/**' -g '!_freeze/**'`
- `scripts/nbv_qmd_outline.sh --compact`

`make qmd-frontmatter-check` remains blocked by pre-existing `docs/contents/ideas.qmd` taxonomy values (`audience: internal`, `status: archive`) outside this cleanup scope.

## Canonical Impact

No canonical state update is needed. The current durable project state already says raw scratch/history belongs under `.agents/archive/docs/` and public docs should expose only curated, current surfaces.

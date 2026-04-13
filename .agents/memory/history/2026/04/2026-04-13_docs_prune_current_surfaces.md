---
id: 2026-04-13_docs_prune_current_surfaces
date: 2026-04-13
title: "Docs Prune Current Surfaces"
status: done
topics: [docs, quarto, navigation, pruning]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/index.qmd
  - docs/_quarto.yml
  - docs/contents/todos.qmd
  - docs/contents/roadmap.qmd
  - docs/contents/impl/aria_nbv_package.qmd
artifacts:
  - docs/_site/index.html
  - docs/_site/contents/todos.html
  - docs/_site/contents/roadmap.html
  - docs/_site/contents/impl/aria_nbv_package.html
assumptions:
  - ideas.qmd is the preferred living future-work surface.
  - The old TODO and roadmap pages should be pruned rather than preserved as long-form main-nav documents.
---

## Task

Prune the docs around current surfaces by collapsing `todos.qmd`, collapsing `roadmap.qmd`, turning `aria_nbv_package.qmd` into a compatibility stub, and simplifying the homepage and Quarto navigation.

## Method

- Replaced `docs/contents/todos.qmd` with a compact `Current Priorities` page that points readers to architecture, model, and ideas docs instead of retaining historical bug/migration logs.
- Replaced `docs/contents/roadmap.qmd` with a short `Project Status` page that captures current state and decision gates instead of the old phase-by-phase thesis timeline.
- Replaced `docs/contents/impl/aria_nbv_package.qmd` with a short compatibility stub and removed it from the main sidebar.
- Simplified `docs/index.qmd` around current architecture pages, current model/training pages, and `ideas.qmd`.
- Updated `docs/_quarto.yml` so the navbar and sidebar emphasize architecture, model docs, ideas, and compact state pages instead of the old TODO/roadmap/package hierarchy.

## Verification

- `cd docs && quarto render .`
- `cd docs && quarto check`
- Spot-checked the generated HTML for:
  - `Current Priorities`
  - `Project Status`
  - `aria_nbv Package (Legacy Stub)`
  - updated navbar labels `Architecture`, `Model`, and `Ideas`

Both commands succeeded. Quarto render still emitted unrelated pre-existing warnings in literature and mirrored agent-scaffold pages.

## Canonical state impact

- None. This was a docs pruning and navigation cleanup pass only.

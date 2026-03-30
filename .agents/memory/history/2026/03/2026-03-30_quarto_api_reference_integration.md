---
id: 2026-03-30_quarto_api_reference_integration
date: 2026-03-30
title: "Integrate quartodoc-generated aria_nbv API reference into Quarto Pages"
status: done
topics: [docs, quarto, github-pages, api-reference]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - docs/_quarto.yml
  - .github/workflows/quarto-publish.yml
  - scripts/quarto_generate_api_docs.sh
  - scripts/validate_agent_memory.py
  - docs/reference/index.qmd
  - docs/reference/_api_index.md
  - docs/reference/_styles-quartodoc.css
  - docs/reference/.gitignore
  - docs/index.qmd
  - docs/contents/impl/aria_nbv_package.qmd
  - docs/AGENTS.md
  - .agents/memory/state/DECISIONS.md
artifacts:
  - docs/reference/*.qmd (generated, ignored)
  - docs/_site/reference/*
---

## Task

Integrate auto-generated `aria_nbv` API docs into the Quarto site and ensure the
GitHub Pages publish workflow refreshes them before rendering.

## Method

Added a `quartodoc` block to `docs/_quarto.yml`, created a small
`scripts/quarto_generate_api_docs.sh` wrapper, added a tracked API landing page
with tracked stub include/CSS files, and updated the Pages workflow to install
`quartodoc`, generate the reference pages, and then run `quarto render docs
--no-execute`. Restored the missing `scripts/validate_agent_memory.py` so the
repo's memory hygiene target works again after adding the debrief and canonical
state update.

## Findings

- `quartodoc` works against the source tree without installing `aria_nbv` when
  `source_dir` points at `../aria_nbv`.
- Global `include_imports: true` caused alias-resolution failures inside
  concrete VIN modules; restricting import following to package-root modules
  fixed the build while preserving re-exported public surfaces.
- The generator emits a few non-fatal warnings from existing docstrings and one
  stale parameter note in `configs/path_config.py`; these do not block publish.

## Verification

- `./scripts/quarto_generate_api_docs.sh`
- `bash -n scripts/quarto_generate_api_docs.sh`
- `quarto render docs --no-execute`
- `quarto check`
- `make check-agent-memory`

## Canonical State Impact

Updated `.agents/memory/state/DECISIONS.md` to record that the published Quarto
site now refreshes `aria_nbv` API reference pages from docstrings during the
Pages workflow.

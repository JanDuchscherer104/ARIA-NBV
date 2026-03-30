---
id: 2026-03-30_quarto_agent_scaffold_pages
date: 2026-03-30
title: "Publish maintained agent scaffold markdown through Quarto Resources"
status: done
topics: [docs, quarto, scaffold, github-pages]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - docs/_quarto.yml
  - docs/index.qmd
  - docs/contents/resources.qmd
  - docs/AGENTS.md
  - .github/workflows/quarto-publish.yml
  - scripts/quarto_generate_agent_docs.py
  - docs/contents/resources/agent_scaffold/*.qmd
  - .agents/memory/state/DECISIONS.md
artifacts:
  - docs/contents/resources/agent_scaffold/**
---

## Task

Publish the maintained agent instructions and scaffold markdown surfaces through
the Quarto site under `Resources`, with generation driven from the canonical
source files rather than hand-maintained copies.

## Method

Added a Python generator that mirrors the maintained scaffold markdown files
into Quarto pages under `docs/contents/resources/agent_scaffold/`, rewrites
their internal links so they resolve inside the site, and emits an overview
index page. Wired those pages into `docs/_quarto.yml`, updated the Pages
workflow to regenerate them before rendering, and refreshed the docs copy that
previously said agent memory was unpublished.

## Findings

- Quarto can render a markdown file outside the docs project as a standalone
  page, but those pages do not land cleanly inside the website output tree for
  navigation.
- Thin generated wrapper pages are the reliable way to publish canonical
  scaffold markdown while preserving the original files as the source of truth.
- Link rewriting is necessary because many scaffold docs refer to repo-root or
  out-of-tree paths that are not directly navigable from a page inside
  `docs/contents/`.

## Verification

- `python3 -m py_compile scripts/quarto_generate_agent_docs.py`
- `./scripts/quarto_generate_agent_docs.py`
- `quarto render docs --no-execute`
- `make check-agent-memory`

## Canonical State Impact

Updated `.agents/memory/state/DECISIONS.md` to record that the published Quarto
site now regenerates the maintained agent scaffold pages from canonical
markdown sources during the Pages workflow.

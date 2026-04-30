---
id: 2026-04-30_shared_glossary_source
date: 2026-04-30
title: "Shared Glossary Source"
status: done
topics: [docs, glossary, typst, quarto, kg]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - docs/glossary/terms.yml
  - scripts/glossary_build.py
  - docs/contents/glossary.qmd
  - docs/contents/questions.qmd
  - docs/typst/shared/glossary.generated.typ
  - docs/_generated/context/glossary.jsonl
  - .configs/litkg.toml
  - Makefile
---

## Task

Implement a shared glossary source for ARIA-NBV terminology that can feed
Quarto docs, Typst paper/slides, and litkg/KG ingestion.

## Method

Added `docs/glossary/terms.yml` as the canonical hand-edited glossary source
and `scripts/glossary_build.py` as the generator behind `make glossary`.
Generated Quarto include content, Typst helper functions, and JSONL concept
records from the same source. The public glossary page now includes generated
content, and `questions.qmd` links to canonical glossary anchors.

## Verification

- `make glossary`
- `cd aria_nbv && uv run ruff format ../scripts/glossary_build.py`
- `cd aria_nbv && uv run ruff check ../scripts/glossary_build.py`
- `cd docs && quarto render contents/glossary.qmd`
- `cd docs && quarto render contents/questions.qmd`
- `cd docs && typst compile typst/paper/main.typ --root .`
- `scripts/nbv_qmd_outline.sh --compact`
- `make kg-materialize`
- `make check-agent-memory`

## Canonical State Impact

`PROJECT_STATE.md` now records the shared glossary as the durable terminology
source and directs thesis docs to link glossary anchors instead of redefining
terms inline.

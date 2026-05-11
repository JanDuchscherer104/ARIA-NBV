---
id: 2026-05-11_impl_docs_to_docstrings
date: 2026-05-11
title: "Implementation Docs Moved To Docstring Contracts"
status: done
topics: [docs, docstrings, quartodoc, context, uml]
confidence: high
canonical_updates_needed: []
---

## Task

Deprecate the public `docs/contents/impl` surface and move useful active
implementation contracts into package docstrings and generated Quartodoc API
pages.

## Method

- Updated the repo-local Python docstring guidance to prefer Quarto-compatible
  Markdown math and backtick references instead of Sphinx/RST roles.
- Added dense module/class contract docstrings for active data-handling,
  candidate/rollout, rendering, RRI, and VIN surfaces.
- Moved the old implementation QMD/Mermaid sources to `.agents/archive/docs/impl/`
  and removed them from public Quarto navigation.
- Extended Quartodoc sections for implementation contracts and repaired the API
  docs helper so tracked reference entrypoints are preserved.
- Added `scripts/filter_mermaid.py` and introduced a `VinOfflineDatasetItem`
  type alias so `make context-uml` can complete.

## Verification

- `make kg-status`
- `make context`
- `make context-uml`
- `make context-docstrings`
- `make api-docs`
- `make qmd-frontmatter-check`
- `cd docs && quarto render reference/index.qmd`
- `cd docs && quarto render contents/thesis/questions.qmd`
- `cd docs && quarto render contents/thesis/roadmap.qmd`
- `cd docs && quarto render contents/glossary.qmd`
- `cd aria_nbv && uv run ruff check <touched Python files>`
- `cd aria_nbv && uv run pytest tests/data_handling/test_public_api_contract.py tests/rri_metrics tests/pose_generation`
- `make agents-db AGENTS_ARGS='validate'`

## Notes

The explicit grep for old impl paths still has expected hits in historical
archives, transcripts, and generated KG evidence. Active public docs, active
skills/references, and package docstrings were updated away from the retired
surface.

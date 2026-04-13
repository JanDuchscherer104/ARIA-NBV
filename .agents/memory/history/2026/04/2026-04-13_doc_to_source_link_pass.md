---
id: 2026-04-13_doc_to_source_link_pass
date: 2026-04-13
title: "Doc To Source Link Pass"
status: done
topics: [docs, quarto, typst, source-links]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - docs/AGENTS.md
  - .agents/memory/state/DECISIONS.md
  - docs/typst/shared/macros.typ
  - docs/contents/impl/aria_nbv_overview.qmd
  - docs/contents/repo_structure.qmd
  - docs/contents/impl/data_pipeline_overview.qmd
  - docs/contents/impl/oracle_rri_impl.qmd
  - docs/contents/impl/coral_intergarion.qmd
  - docs/contents/impl/vin_nbv.qmd
  - docs/contents/impl/aria_nbv_package.qmd
  - docs/typst/paper/sections/05-oracle-rri.typ
  - docs/typst/paper/sections/06-architecture.typ
  - docs/typst/paper/sections/12f-appendix-pose-frames.typ
artifacts:
  - docs/_generated/context/source_index.md
  - docs/_generated/context/literature_index.md
  - docs/_generated/context/data_contracts.md
assumptions:
  - GitHub source links should target the repository `main` branch.
  - Sparse source anchors are preferable to dense inline linking for implementation docs.
---

## Task

Implement a doc-to-source link pass across the current implementation docs and matching Typst paper sections, while fixing stale ownership references and standardizing the source-linking convention.

## Method

- Refreshed lightweight routing artifacts with `make context`, using the local `uv` Python 3.11 runtime via `PYTHON_INTERPRETER=...` because the default repo venv was unavailable.
- Added the Quarto-side `Source anchors` convention to `docs/AGENTS.md` and recorded the workflow rule in canonical decisions.
- Updated the selected implementation-heavy Quarto pages to point at the current `app`, `data_handling`, `pipelines`, `rri_metrics`, `lightning`, and `vin` owners.
- Extended the Typst `#gh(...)` helper to support optional line anchors and explicit labels while keeping one-argument calls backward-compatible.
- Added high-signal GitHub source anchors to the current-code Typst sections for oracle labeling and VIN.

## Verification

- `make context PYTHON_INTERPRETER=/Users/jd/.local/share/uv/python/cpython-3.11.14-macos-aarch64-none/bin/python3.11`
- `python3.11 scripts/quarto_generate_agent_docs.py`
- `cd docs && quarto render .`
  - Required creating a local `aria_nbv` Jupyter kernelspec because Quarto would not resolve the project without that kernel name.
  - Render completed with pre-existing unrelated warnings in literature/agent-scaffold pages.
- `cd docs && typst compile typst/paper/main.typ --root .`
  - Succeeded with an existing font warning for `DejaVu Serif`.
- `cd docs && typst compile typst/slides/slides_thesis_outlook.typ --root .`
  - Succeeded with existing font warnings for `Open Sans`.
- `rg` verification on the edited pages confirmed the retired internal paths targeted by the task were removed.

## Canonical state impact

- Updated `.agents/memory/state/DECISIONS.md` with the new rule for `Source anchors` callouts and inline `blob/main#Lx` usage in implementation-focused Quarto pages.

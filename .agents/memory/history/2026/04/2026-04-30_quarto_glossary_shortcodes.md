---
id: 2026-04-30_quarto_glossary_shortcodes
date: 2026-04-30
title: "Quarto Glossary Shortcodes"
status: done
topics: [docs, glossary, quarto, kg]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - docs/_extensions/aria-glossary/
  - scripts/glossary_build.py
  - docs/contents/glossary.qmd
  - docs/contents/thesis/questions.qmd
  - docs/contents/thesis/roadmap.qmd
  - docs/contents/theory/rri_theory.qmd
---

## Task
Implemented Quarto-native glossary shortcodes so QMD prose can reference
canonical terms from `docs/glossary/terms.yml` without hand-written glossary
links.

## Method
Added the local `aria-glossary` shortcode extension, generated a Lua term map
from `make glossary`, and registered the shortcode file in `docs/_quarto.yml`.
The public syntax is `{{< gls term-id >}}` for short labels and
`{{< glsfull term-id >}}` for full labels. Unknown terms render visibly and
emit a Quarto warning.

## Outputs
Migrated selected thesis question, roadmap, and RRI theory references to the new
shortcodes. Updated the glossary page authoring note and canonical project
state to name the shortcode workflow.

## Verification
Ran `make glossary`, `ruff format/check` on the generator, targeted Quarto
renders for the glossary, thesis questions, thesis roadmap, and RRI theory,
`scripts/nbv_qmd_outline.sh --compact`, `make kg-materialize`, and
`make check-agent-memory`.

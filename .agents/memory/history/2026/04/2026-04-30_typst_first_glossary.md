---
id: 2026-04-30_typst_first_glossary
date: 2026-04-30
title: "Typst-First Glossary"
status: done
topics: [docs, typst, quarto, glossary, kg]
confidence: high
canonical_updates_needed: []
---

## Task

Move the shared ARIA-NBV glossary edit point from YAML to Typst while preserving
the Quarto shortcode surface and KG export.

## Method

Added `docs/typst/shared/glossary.typ` as the canonical glossarium-backed source,
with stable term IDs and machine-readable `custom` metadata queried via
`typst query`. Updated `scripts/glossary_build.py` so `make glossary` generates
the Quarto glossary page, Typst helper facade, Lua shortcode data, compatibility
YAML, and JSONL from the Typst source.

Added `docs/typst/glossary/main.typ` as a standalone inspection document using
`@preview/glossarium:0.5.10`.

## Verification

- `make glossary`
- `cd docs && typst compile typst/glossary/main.typ --root .`
- `cd docs && typst compile typst/seminar_paper/main.typ --root .`
- `cd docs && quarto render contents/glossary.qmd`
- `cd docs && quarto render contents/thesis/questions.qmd`
- `scripts/nbv_qmd_outline.sh --compact`
- `make kg-materialize`

The combined two-file Quarto render form was not used because this Quarto
version treated the second QMD path as an output target; rendering the pages
separately succeeded.

## Canonical State Impact

The durable convention is encoded in `PROJECT_STATE.md`, the new Typst source
comments, generated glossary intro, and litkg source configuration.

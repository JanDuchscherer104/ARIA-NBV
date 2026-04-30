---
id: 2026-04-30_typst_layout_split
date: 2026-04-30
title: "Split Typst seminar and thesis document layout"
status: done
topics: [docs, typst, seminar, thesis]
confidence: high
canonical_updates_needed:
  - AGENTS.md
  - docs/AGENTS.md
  - .agents/memory/state/DECISIONS.md
---

## Task

Move the Typst seminar paper and slide decks into explicit seminar/thesis layout directories, and move shared slide infrastructure out of the old slides directory.

## Method

Used `git mv` for the directory and file moves. The seminar paper now lives under `docs/typst/seminar_paper/`, seminar decks under `docs/typst/seminar_slides/`, the thesis outlook deck under `docs/typst/thesis_slides/`, and the shared slide template plus slide data under `docs/typst/shared/`.

Updated current path contracts in repo guidance, docs guidance, Quarto navigation, Makefile defaults, KG ingestion defaults, active skills/scripts, and glossary source links. Regenerated glossary artifacts after updating `docs/glossary/terms.yml`.

## Verification

- `make glossary` passed.
- `cd docs && typst compile typst/seminar_paper/main.typ /tmp/aria-nbv-seminar-paper.pdf --root .` passed.
- `cd docs && typst compile typst/seminar_slides/slides_4.typ /tmp/aria-nbv-seminar-slides-4.pdf --root .` passed with the existing Open Sans warning.
- `cd docs && typst compile typst/thesis_slides/slides_thesis_outlook.typ /tmp/aria-nbv-thesis-outlook.pdf --root .` passed with the existing Open Sans warning.
- `make typst-paper`, default `make typst-slide`, and thesis slide compilation through `make typst-slide SLIDES=docs/typst/thesis_slides/slides_thesis_outlook.typ` passed.

## Notes

`slides_1.typ` still fails on missing `docs/figures/scene-script/ase_modalities.jpg`; this was not introduced by the move because the deck remains at the same directory depth relative to `docs/figures/`.

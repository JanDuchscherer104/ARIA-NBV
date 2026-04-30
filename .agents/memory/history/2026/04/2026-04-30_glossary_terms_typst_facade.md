---
id: 2026-04-30_glossary_terms_typst_facade
date: 2026-04-30
title: "Generate Typst term constants from shared glossary"
status: done
topics: [docs, typst, glossary, notation]
confidence: high
canonical_updates_needed: []
---

## Task

Collapse the hand-maintained `docs/typst/shared/terms.typ` acronym list into the shared glossary source.

## Method

Added optional `typst_macro` metadata to `docs/glossary/terms.yml` and extended `scripts/glossary_build.py` so `docs/typst/shared/glossary.generated.typ` emits backwards-compatible `#Name` and `#Name_full` constants. Converted `docs/typst/shared/terms.typ` into a compatibility facade that imports the generated glossary Typst artifact.

Added glossary records for term constants that previously existed only in `terms.typ`: CR, CD, DoF, 6DoF, 5DoF, AUC, MVS, Occupancy Grid, ADT, and AEO. Existing glossary records now own RRI, NBV, GT, PC, SLAM, ASE, EFM3D, EVL, and SSL Typst macro exports.

## Verification

- `make glossary` passed and validated 38 glossary terms.
- `cd docs && typst compile typst/seminar_paper/main.typ /tmp/aria-nbv-seminar-paper-terms.pdf --root .` passed.
- `cd docs && typst compile typst/seminar_slides/slides_4.typ /tmp/aria-nbv-seminar-slides-4-terms.pdf --root .` passed with the existing Open Sans warning.
- `cd docs && typst compile typst/thesis_slides/slides_thesis_outlook.typ /tmp/aria-nbv-thesis-outlook-terms.pdf --root .` passed with the existing Open Sans warning.

## Notes

The active Typst paper directory in the worktree is `docs/typst/seminar_paper/`; stale `seminar_pape` references were aligned to that actual path while validating this change.

---
id: 2026-05-15_typst_glossarium_first_use_facade
date: 2026-05-15
title: "Typst Glossarium First-Use Facade"
status: done
topics: [docs, typst, glossary]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/shared/glossary.typ
  - docs/typst/shared/macros.typ
  - docs/typst/thesis/advisor_distillation.typ
  - docs/typst/thesis/main.typ
  - docs/typst/thesis/proposal.typ
  - docs/typst/thesis/sections/01-introduction.typ
  - docs/typst/thesis/sections/02-background.typ
  - docs/typst/thesis/sections/proposal/01-motivation.typ
  - docs/typst/thesis/sections/proposal/02-problem.typ
  - docs/typst/thesis/sections/proposal/02-related-work.typ
  - docs/typst/thesis/sections/proposal/03-objectives.typ
  - docs/typst/thesis/advisor_distillation.pdf
  - docs/typst/thesis/proposal.pdf
---

## Task

Implemented Glossarium first-use long-short behavior for ARIA-NBV Typst thesis and proposal surfaces.

## Method

Updated `docs/typst/shared/glossary.typ` to alias Glossarium's `gls` and `glspl` functions and expose ARIA wrappers with `link: false` by default, because handouts and proposal PDFs do not print glossary targets. Updated `docs/typst/shared/macros.typ` so existing section imports can use the live `#gls(...)` facade. Registered the glossary in thesis/proposal entrypoints and replaced selected prose acronyms with glossary references while leaving compact table and metric contexts short.

## Verification

- `/tmp` fixture compiled and extracted as `Relative Reconstruction Improvement (RRI) and RRI.`
- `cd docs && typst compile typst/thesis/advisor_distillation.typ typst/thesis/advisor_distillation.pdf --root .`
- `cd docs && typst compile typst/thesis/proposal.typ typst/thesis/proposal.pdf --root .`
- `cd docs && typst compile typst/thesis/main.typ /tmp/thesis-main-glossarium.pdf --root .`
- `cd docs && typst compile typst/glossary/main.typ /tmp/aria-glossary-main.pdf --root .`
- Rendered advisor and proposal opening pages to PNG and visually checked first-use expansion.
- Typst hygiene strict checks passed with only existing advisory glossary/notation prompts.

## Canonical State Impact

No project-truth or thesis-direction update is needed. This is a Typst authoring behavior change only.

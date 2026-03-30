---
scope: paper
applies_to: docs/typst/paper/**
summary: Paper narrative, figure, and citation synchronization guidance for work under docs/typst/paper/.
---

# Paper Boundary

Apply this file when working under `docs/typst/paper/`.

## Public Contracts
- Canonical paper entrypoint: `docs/typst/paper/main.typ`
- Section sources: `docs/typst/paper/sections/`
- Paper-local figures and data: `docs/typst/paper/figures/`, `docs/typst/paper/data/`
- Shared bibliography: `docs/references.bib`
- Alignment surfaces outside the paper: `docs/contents/`, `docs/typst/slides/`

## Boundary Rules
- `docs/typst/paper/main.typ` is the highest-level project truth for research narrative, system description, and thesis-facing claims.
- Keep Quarto docs and slides aligned to the paper when a change affects terminology, system behavior, experimental claims, or figure interpretation.
- Prefer editing `.typ`, figure, and data sources instead of compiled artifacts such as `docs/typst/paper/main.pdf`.
- Keep citations, figure references, and terminology synchronized across paper sections and `docs/references.bib`.
- Do not introduce placeholder citations or paper-only terminology that conflicts with the implementation and Quarto surfaces.

## Verification
- Run `cd docs && typst compile typst/paper/main.typ --root .` for paper text, figure, or citation changes.
- Run the relevant Quarto render or slide compile if the same narrative, figures, or terminology are shared outside the paper.
- Confirm bibliography entries, figure references, and section cross-references remain consistent after substantive edits.

## Completion Criteria
- The paper compiles successfully from `main.typ`.
- Citations, terminology, and figure references are synchronized.
- Related Quarto or slide surfaces were updated when the paper change altered shared narrative or claims.

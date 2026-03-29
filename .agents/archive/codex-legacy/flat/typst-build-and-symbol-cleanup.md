# Typst build + symbol cleanup (paper/slides)

Date: 2026-01-29

## Build gotchas

- In this environment, `typst` may resolve to the snap wrapper (`/snap/bin/typst`) and fail with `snap-confine ... cap_dac_override`. Use the raw binary instead:
  - `/snap/typst/current/bin/typst`
- Typst project root for the paper/slides is `docs` so that absolute asset paths resolve:
  - `--root docs` (so `/figures/...`, `/references.bib` point to `docs/figures/...`, `docs/references.bib`)

## Fixes applied

- `docs/typst/paper/sections/04-dataset.typ`: added missing `#import "../../shared/macros.typ": *` (required for `#ASE_full`, `#ASE`).
- `docs/typst/paper/sections/05-coordinate-conventions.typ`: fixed math example to avoid `#T(...)` inside `$...$` (Typst treats this as invalid code); use `T(symb.frame.cq, symb.frame.w)` instead.
- `docs/typst/paper/sections/05-oracle-rri.typ`: removed informal arrow notation (`P → M`, `M → P`) and referenced the named accuracy/completeness terms via `#symb.oracle.acc (...)` / `#symb.oracle.comp (...)`.
- `docs/typst/paper/main.typ`: re-enabled `sections/12f-appendix-pose-frames.typ` because it is referenced from diagnostics.
- `docs/typst/paper/sections/07-training-objective.typ`: replaced markdown-style `**bold**` with Typst emphasis (`*...*`) to avoid warnings.

## Sanity compile commands

- Paper:
  - `/snap/typst/current/bin/typst compile --root docs docs/typst/paper/main.typ /tmp/nbv_paper.pdf`
- Slides:
  - `/snap/typst/current/bin/typst compile --root docs docs/typst/slides/slides_4.typ /tmp/slides_4.pdf`


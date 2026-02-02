# Makefile Typst targets (2026-01-24)

## Summary
- Added Makefile targets to compile the Typst paper and any slide deck via `typst`.
- Default slide target compiles `slides_4.typ`, but any slide file can be specified with `SLIDES=...`.

## New targets
- `make typst-paper` -> compiles `docs/typst/paper/main.typ` to `docs/typst/paper/main.pdf`.
- `make typst-slide SLIDES=slides_4.typ` -> compiles slide deck to a matching PDF in `docs/typst/slides/`.

## Notes
- Targets use `--root docs` so `/figures/...` and `/references.bib` resolve reliably.
- You can pass a full path in `SLIDES` (e.g. `docs/typst/slides/slides_2.typ`).

## Verification
- Not executed as part of this change.

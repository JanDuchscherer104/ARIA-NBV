# Typst Paper Guidance

This file applies to `docs/typst/paper/` and adds paper-specific deltas on top
of [../AGENTS.md](../../AGENTS.md).

## Sources Of Truth
- `main.typ`: paper entry point and include order.
- `sections/`: manuscript sections and appendices.
- `../shared/`: shared macros used by paper and slides.
- `docs/references.bib`: bibliography source.

## Rules
- Preserve the current paper architecture unless the task explicitly changes it.
- Keep final manuscript text in full paragraphs, not bullet lists.
- Keep terminology aligned with Quarto docs and canonical state.
- Use Typst cross-references and bibliography keys rather than manual labels in
  prose.
- Prefer existing macros and shared styles before adding new local formatting.

## Verification
- Use `scripts/nbv_typst_includes.py --paper --mode outline` before broad paper
  edits.
- For meaningful paper changes, compile with
  `cd docs && typst compile typst/paper/main.typ --root .`.

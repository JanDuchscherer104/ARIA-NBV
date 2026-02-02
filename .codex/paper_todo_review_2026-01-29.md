# Paper TODO review (2026-01-29)

## Summary
- Added dashy-todo annotations across all paper sections, marking issues and verification points.
- Inserted a TODO outline in `main.typ` so all `#todo` items are discoverable.
- Added TODOs to every appendix section, including the new slide-figure appendix.

## Files touched (high level)
- `docs/typst/paper/main.typ` (dashy import + TODO outline)
- All `docs/typst/paper/sections/*.typ` (inline `#todo[...]`)

## Compile
- `typst compile docs/typst/paper/main.typ /tmp/paper.pdf --root docs`
  - Warning: layout did not converge within 5 attempts (likely due to `#outline` + todo figures).

## Follow-ups
- Inspect the PDF for any layout artifacts caused by the TODO outline.
- If layout warning persists, consider moving the TODO outline to an appendix-only page or reducing the number of inline TODOs per page.

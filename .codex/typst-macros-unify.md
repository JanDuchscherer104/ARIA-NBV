## Summary
- Unified Typst macros into a single shared file for both paper and slides.

## Changes
- Moved utility helpers (`term`, `filepath`, `paperref`) into `docs/typst/shared/macros.typ`.
- Updated paper imports to `../shared/macros.typ` (including `docs/typst/paper/main.typ` and section files).
- Removed duplicate `docs/typst/paper/macros.typ`.

## Follow-ups
- If needed, run `typst compile` for paper/slides to validate includes.

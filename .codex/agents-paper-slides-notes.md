# Paper + Slides AGENTS Notes

## Task summary
- Created a specialized agent guide for Typst paper/slides: `.codex/AGENTS-paper-slides.md`.
- Included template usage, slide/paper conventions, Typst quick reference, and adapted cross-project guidelines.
- Added a research-backed academic writing checklist section and extracted sources into `.codex/academic-writing-guidelines.md`.
- Refactored `.codex/AGENTS-paper-slides.md` to be shorter, removed inline TODOs, and added a “Core Reference Papers” table mapping bib keys ↔ local LaTeX sources ↔ literature QMD pages.
- Added an explicit “Compile → Inspect → Critique → Fix” loop with proven commands for PDF/PNG rendering and visual QA.

## Verified commands (Typst 0.14.1)
- Slides compile + PNG export works with `--root docs` when writing outputs into `.codex/_render/`.
- Paper compile initially failed due to missing `docs/_shared/references.bib`; creating `docs/_shared/` and copying `docs/references.bib` resolves it.
- PDF → PNG fallback works via `pdftoppm` and `mutool draw` (both present in this environment).

## Known caveats
- Typst output to `/tmp/...` failed in this environment (PDF write error). Use repo-local output paths.
- `@preview/charged-ieee:0.1.4` warns about missing `TeX Gyre Termes` and `TeX Gyre Cursor`. It still compiles (likely font fallback), but consider addressing for camera-ready output.

## Key findings
- Paper template is `@preview/charged-ieee:0.1.4` with `#show: ieee.with(...)` in `docs/typst/paper/main.typ`.
- Slides use `@preview/definitely-not-isec-slides:1.0.1` and often `@preview/muchpdf:0.1.1`.
- Shared Typst macros and symbols live in `docs/typst/shared/macros.typ`.

## Potential issues / follow-ups
- `docs/typst/paper/main.typ` points to `/_shared/figures/`, but no `_shared/` directory exists under `docs/typst/paper/` or `docs/typst/`. Confirm intended location and create if needed.
- Decide whether new slides should adopt `custom-template.typ` for rounded `color-block` styling (currently only available, not used in slides_1/2).

## Suggestions
- Keep `docs/references.bib` as the single bibliography source for paper and slides.
- When adding figures, standardize on the existing `fig_path` pattern per deck.
- Run `typst compile` for any edited deck or paper section before finalizing.

## Tests
- No tests were run (documentation-only change).

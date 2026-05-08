# Workflow: Typst Edit Loop

Use this loop for non-trivial changes (multi-paragraph edits, layout changes, or figures).

## Visuals-first loop (tables, equations, diagrams)
When building complex visuals, work in isolation to reduce noise and make issues obvious.

1) Create a minimal `.typ` that only contains the object.
2) Render to PNG for fast inspection (use `typst compile --format png`).
3) Critique the PNG, fix issues, and repeat until clean.
4) Integrate into the main document and re-check in context.

### PNG export commands (CLI)

- **Helper script (preferred):**
  - `.codex/skills/typst-authoring/scripts/render_png.sh -i input.typ -o out --ppi 300 --pages 1`
- **Single page to PNG (explicit):**
  - `typst compile input.typ output.png --format png --ppi 300 --pages 1`
- **Multi-page to PNG (template required):**
  - `typst compile input.typ 'out/{0p}.png' --format png --ppi 300`
- **High-detail inspection (larger PPI):**
  - `typst compile input.typ 'out/{0p}.png' --format png --ppi 600`

Notes:
- Default PPI is `144`. Use `300`–`600` for clean inspection, `1200` for print-quality diagrams.
- If you need transparency, set `#set page(fill: none)` in the `.typ` file.
- PNG text is not extractable; keep PDFs for accessibility.

## Compile → Inspect → Critique → Fix
1) Compile to PDF (no errors; warnings reviewed).
2) Inspect the relevant pages (PDF or rendered PNGs).
3) Critique with the checklist below and fix issues.
4) Re-compile and re-inspect until clean.

## Critique Checklist (fast)
- **Text:** no overflow/cutoff, consistent sizes, readable line lengths.
- **Layout:** clean alignment, balanced grids, no orphaned elements.
- **Figures:** correct scale/cropping, legible labels, consistent captions.
- **Equations:** rendered correctly, consistent notation, referenced.
- **References:** citations resolved; cross-refs point to correct objects.

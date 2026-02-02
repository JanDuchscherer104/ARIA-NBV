# Task Notes: Streamlit Figures in Typst Paper (2026-01-24)

## Scope
- Enriched the Typst paper with Streamlit app figures and reduced reliance on generic figures.
- Focused on sections: dataset, architecture, system pipeline, diagnostics.

## Key Changes
- Added Streamlit diagnostics grid in `docs/typst/paper/sections/04-dataset.typ` and updated text to reference the in-paper overlay.
- Added an EVL occupancy-prior diagnostic figure (`scene_field_occ_pr.png`) in `docs/typst/paper/sections/06-architecture.typ`.
- Replaced the single pipeline figure in `docs/typst/paper/sections/08-system-pipeline.typ` with a 2x2 Streamlit diagnostics grid.
- Inserted a candidate-sampling diagnostics grid in `docs/typst/paper/sections/09-diagnostics.typ` and updated the appendix note.
- Updated `.codex/AGENTS_INTERNAL_DB.md` to document the Streamlit figure export script.
- Mirrored `rri_hist_81056_000022.png` into `docs/figures/app/` to satisfy Typst `/figures/...` resolution.

## Files Touched
- `docs/typst/paper/sections/04-dataset.typ`
- `docs/typst/paper/sections/06-architecture.typ`
- `docs/typst/paper/sections/08-system-pipeline.typ`
- `docs/typst/paper/sections/09-diagnostics.typ`
- `.codex/AGENTS_INTERNAL_DB.md`
- `docs/figures/app/rri_hist_81056_000022.png`

## Follow-ups / Suggestions
- If any figures need regeneration, run:
  `uv run python oracle_rri/scripts/export_paper_figures.py --config-path .configs/paper_figures_oracle_labeler.toml --output-dir docs/figures/app --overwrite`.
- Consider moving or de-emphasizing remaining generic figures (e.g., VIN-NBV diagram, ATEK overview) if the paper should be fully Streamlit-first.

## Testing
- `typst compile docs/typst/paper/main.typ --root docs` (requires elevated permissions due to snap packaging).

# slides_4.typ revision notes (2026-01-24)

## Summary
- Added a new slide deck at `docs/typst/slides/slides_4.typ` aligned to the paper.
- Integrated real figures from Streamlit diagnostics (`docs/figures/app`), Optuna sweep plots (`docs/figures/vin_v2`), and W&B snapshots (`docs/figures/vin_v2/wandb_loss_lr_T20_vs_T30.png`).
- Included the W&B run summary table from `docs/typst/paper/sections/09c-wandb.typ` and key oracle RRI distribution stats from `docs/typst/paper/sections/05-oracle-rri.typ`.
- Switched the deck to use the shared `docs/typst/shared/template.typ` (and `notes.typ`) for the slide overrides.

## Decisions
- Kept the slide theme consistent with `slides_2.typ` / `slides_3.typ` (definitely-not-isec template + shared macros).
- Used ASCII-only text and Typst symbols in markup/math (no Unicode glyphs).
- Organized sections: Motivation, Data + Oracle pipeline, VIN v2 architecture, Diagnostics, Sweep evidence, Summary.

## Follow-ups
- Optional: compile slides with Typst to validate figure paths and layout.
- Optional: add `slides_4.pdf` and link in `docs/index.qmd` if you want it listed alongside other presentations.
- If you want different Optuna or W&B runs highlighted, specify the trial IDs and I can swap the plots/text.

## Verification
- `typst compile --root docs docs/typst/slides/slides_4.typ /tmp/slides_4.pdf`

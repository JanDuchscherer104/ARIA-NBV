# Slides 4 Revision (Final Presentation)

Date: 2026-01-27

## Goal
Revise the final presentation deck `docs/typst/slides/slides_4.typ` to be:
- implementation-grounded (oracle pipeline + offline cache + VIN v3 baseline),
- less "AI slop" (fewer generic claims, fewer hard-coded numbers),
- aligned with the Typst paper notation (`docs/typst/shared/macros.typ`),
- enriched with real figures from Streamlit diagnostics, Optuna, and local W&B artifacts.

## What Changed
### Slides
- Rewrote `docs/typst/slides/slides_4.typ` into a structured deck:
  - Oracle RRI pipeline (stages + config)
  - Offline cache + batching contract
  - VIN v3 architecture (global context + FiLM, semidense projections)
  - VIN-NBV (Frahm 2025) vs VIN v3 feature differences
  - Failure modes + diagnostics checks (from `docs/contents/todos.qmd` + our analysis notes)
  - Training dynamics (LR/noise/plateau) + mode-collapse case study (vin-v3-01 vs T41)
  - Objective + metrics (CORAL + scheduled coverage weighting + metric definitions)
  - Empirical evidence (Optuna plots + best W&B run summary + confusion-matrix sequences)
  - Limitations + master-thesis next steps
- Replaced manual / inconsistent tables with values imported from JSON artifacts.
- Removed non-ASCII glyphs in slides; use `#sym.*` or ASCII text only (no raw unicode arrows/dots).

### Shared notation/macros
- Updated `docs/typst/shared/macros.typ`:
  - Added dimension symbol `s.Nq` for candidate count.
  - Added reusable equations used in the deck/paper:
    - CORAL loss + marginal conversion + expected-value prediction
    - CORAL relative-to-random baseline, Spearman/top-k/confusion-matrix definitions, candidate validity, grad-norm definition
    - scheduled coverage weighting (+ normalized weighted loss)
    - FiLM modulation
    - semidense projection validity and visibility fraction

## New Data Artifacts (Typst imports)
All numeric values used in slides come from these files:
- `docs/typst/slides/data/offline_cache_stats.json`
  - derived from `/mnt/e/wsl-data/ase-atek-nbv/offline_cache/{metadata.json,index.jsonl,train_index.jsonl,val_index.jsonl}`
  - includes cache size, scene coverage, and oracle labeler config (candidate sampling + depth render settings)
- `docs/typst/slides/data/wandb_rtjvfyyp_summary.json`
  - derived from `.logs/wandb/wandb/run-20260126_205313-rtjvfyyp/files/{wandb-summary.json,vin_effective.json}`
  - includes key end-of-run metrics and the effective VIN v3 config
- `docs/typst/slides/data/wandb_rtjvfyyp_dynamics.json`
  - distilled training-dynamics numbers used in the slides (plateau epoch, LR range, loss trend summaries)
- `docs/typst/slides/data/vin_v3_01_vs_t41_summary.json`
  - distilled comparison used for the mode-collapse case study slide (vin-v3-01 vs T41)
- `docs/typst/paper/data/optuna_v2_top_trials.csv`
  - reused for Optuna evidence in the deck

## New/Curated Figures
- Copied W&B confusion matrices + label histograms for `rtjvfyyp` into:
  - `docs/figures/wandb/rtjvfyyp/train-figures/`
  - `docs/figures/wandb/rtjvfyyp/val-figures/`
  - plus `frames.json` manifests for Typst animation with `conf-matrix-sequence`.
- Added offline cache plots:
  - `docs/figures/offline_cache/scene_coverage.png`
  - `docs/figures/offline_cache/snippets_per_scene_hist.png`

## Compile Command
Typst must be compiled with the project root set to `docs` so that `/figures/...`
and `/references.bib` resolve correctly:

```bash
typst compile --root docs docs/typst/slides/slides_4.typ docs/typst/slides/slides_4.pdf
```

## Notes / Follow-ups
- If we want full training-dynamics plots (loss vs step, lr schedule over time, etc.) without W&B API access,
  we likely need to export histories from the local `run-*.wandb` file into CSV first.
- Consider adding a small `scripts/export_wandb_history_local.py` utility that reads local `.wandb` files
  into a tidy dataframe for consistent plotting into `docs/figures/wandb/...`.

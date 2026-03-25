---
id: 2025-12-31_wandb_analysis_panel_2025-12-31
date: 2025-12-31
title: "Wandb Analysis Panel 2025 12 31"
status: legacy-imported
topics: [wandb, analysis, panel, 2025, 12]
source_legacy_path: ".codex/wandb_analysis_panel_2025-12-31.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# W&B Analysis Panel (2025-12-31)

## Summary
- Added a new Streamlit panel for W&B run analysis with derived diagnostics (trend smoothing, train/val gaps, calibration bias, correlation, volatility) and direct rendering of logged confusion matrices + label histograms.
- Wired the new panel into the app navigation.
- Implemented helper utilities in `panels.py` for W&B run resolution, history loading, metric pairing, and media download.

## Files Touched
- `oracle_rri/oracle_rri/app/panels.py`
- `oracle_rri/oracle_rri/app/app.py`

## Key Design Notes
- Run input accepts full W&B path, run id, or display name; when only an id/name is provided, the entity/project fields are used for resolution.
- Media rendering pulls confusion matrix + label histogram images via W&B history and downloads them into `.logs/wandb/api_media/<run_id>`.
- Derived diagnostics are intended to complement (not duplicate) W&B dashboards.

## Tests / Validation
- `ruff format oracle_rri/oracle_rri/app/panels.py oracle_rri/oracle_rri/app/app.py`
- `ruff check oracle_rri/oracle_rri/app/panels.py oracle_rri/oracle_rri/app/app.py` (fails due to pre-existing lint violations across these files)
- `pytest oracle_rri/oracle_rri/app/panels.py` (fails during import: missing `power_spherical`)

## Follow-ups / Suggestions
- Install `power_spherical` in the active env to allow UI modules to import during pytest collection.
- If desired, add a small config section in the panel to persist default entity/project in `WandbConfig` or app state.
- Consider caching W&B history/media in a dedicated lightweight cache to avoid repeated API calls on rerun.

## Update: RRI Bias + Variance (2025-12-31)
- Extended the W&B panel to report residual bias and variance: mean residual, |bias|, bias^2, residual variance, residual MSE.
- Added rolling variance diagnostics for residuals and for pred/oracle variance curves.

## Notes
- `ruff format oracle_rri/oracle_rri/app/panels.py` failed with: `Unexpected indentation` at ~line 4179, but `python -m py_compile` succeeds. This appears to be a ruff parsing issue in this large file.

## Update: Offline Cache Index Rebuild (2025-12-31)
- Rebuild index now parses `samples/*.pt` filenames instead of loading payloads.
- Extraction assumes `ASE_NBV_SNIPPET_<scene>_<snippet>_<hash>[__suffix]` naming; scene/snippet are derived from the stem.

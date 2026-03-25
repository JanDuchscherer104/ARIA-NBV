---
id: 2026-01-07_optuna_sweep_panel_2026-01-07
date: 2026-01-07
title: "Optuna Sweep Panel 2026 01 07"
status: legacy-imported
topics: [optuna, sweep, panel, 2026, 01]
source_legacy_path: ".codex/optuna_sweep_panel_2026-01-07.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Optuna Sweep Panel (2026-01-07)

## Summary
- Added a new Streamlit panel (`oracle_rri/oracle_rri/app/panels/optuna_sweep.py`) to explore Optuna sweep results from local SQLite studies, including objective vs trial plots and per-parameter effect plots.
- Wired the panel into the app navigation (`oracle_rri/oracle_rri/app/app.py`) and exported it via the panels package (`oracle_rri/oracle_rri/app/panels/__init__.py`).
- Documented the new page in `docs/contents/impl/data_pipeline_overview.qmd`.
- Added tests for helper functions in `tests/app/panels/test_optuna_sweep_panel.py`.

## Notes / Findings
- The panel currently focuses on Optuna DB summaries. W&B cross-run curves remain in the dedicated W&B panel.
- Optuna import is optional; the UI shows a warning if the dependency is missing.

## Follow-ups / Suggestions
- Optional: add Optuna parameter-importance plots using `optuna.importance.get_param_importances` once we decide on a robust importance method for the study.
- Optional: add a W&B overlay selector for a pair of trials (mirroring `.codex/plot_optuna_vin_v2_2026-01-07.py`) to tie optuna params to training dynamics directly in the panel.

## Update (2026-01-07)
- Expanded Optuna sweep panel with tabs for Overview, Parameter effects, Interactions, Importance, Duplicates, and Trial table.
- Added objective distribution (violin) and top-K trial table.
- Added interaction heatmap with numeric binning and aggregation controls.
- Added Optuna parameter-importance plot using `optuna.importance.get_param_importances`.
- Added duplicate-config detector based on a selectable signature of params.
- Added tests for binning and interaction matrix helpers.

## Update (query + width)
- Added sidebar trial filter using pandas query strings (stored in session_state).
- Applied query after state/non-finite filtering with warning on invalid expressions.
- Replaced deprecated `use_container_width` with `width="stretch"` in Optuna sweep panel.

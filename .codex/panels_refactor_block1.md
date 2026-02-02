# Panels refactor block 1 (RRI panel plotting)

Date: 2026-01-26

## Scope
- Moved RRI plotting logic into `oracle_rri/rri_metrics/plotting.py` and updated
  `oracle_rri/app/panels/rri.py` to call those helpers.
- Added unit tests covering the new RRI plotting helpers.

## Findings
- The RRI panel no longer builds plotly figures inline; bar charts and scene
  plotting are now centralized in `rri_metrics.plotting`.
- `plot_rri_scene` uses `RenderingPlotBuilder` and remains mesh-optional; it
  still renders semidense points and selected candidates as before.

## Decisions
- Keep Streamlit UI (selection widgets + popovers) in the panel, but isolate
  all figure construction in the component plotting module.

## Follow-ups
- Continue extracting plotly builders from `app/panels/rri_binning.py`,
  `app/panels/offline_stats.py`, VIN diagnostic tabs, and wandb/optuna panels.

## Tests
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/test_plotting_helpers_refactor.py`

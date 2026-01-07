# Panels refactor subpackage (2026-01-01)

## Summary
- Split `oracle_rri/oracle_rri/app/panels.py` into a new subpackage `oracle_rri/oracle_rri/app/panels/` with page modules (`data.py`, `candidates.py`, `depth.py`, `rri.py`, `vin_diagnostics.py`, `rri_binning.py`, `wandb.py`, `offline_stats.py`) and shared helpers (`common.py`, `plot_utils.py`, `rri_utils.py`, `wandb_utils.py`, `doc_classifier_utils.py`, `vin_utils.py`, `offline_cache_utils.py`).
- Removed legacy `oracle_rri/oracle_rri/app/panels.py` to avoid module/package name collision; `app.py` now imports from `oracle_rri.app.panels` package exports.
- Updated `docs/contents/todos.qmd` references to `panels/` subpackage.

## Tests / Verification
- `ruff format oracle_rri/oracle_rri/app/panels` and `ruff check oracle_rri/oracle_rri/app/panels` pass.
- `oracle_rri/.venv/bin/python -m pytest tests` fails during collection with:
  - `ImportError: cannot import name 'backproject_depth' from oracle_rri.rendering.unproject` (pre-existing issue).

## Follow-ups / Suggestions
- Fix or restore `backproject_depth` in `oracle_rri/rendering/unproject.py` or adjust tests expecting it.
- If you want generated docs updated, re-run Quarto build to refresh `docs/contents/todos.html` and `docs/search.json`.

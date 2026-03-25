---
id: 2026-01-01_wandb_analysis_app_panels_2026-01-01
date: 2026-01-01
title: "Wandb Analysis App Panels 2026 01 01"
status: legacy-imported
topics: [wandb, analysis, app, panels, 2026]
source_legacy_path: ".codex/wandb_analysis_app_panels_2026-01-01.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# W&B analysis page update (app panels)

## Summary
- Replaced `render_wandb_analysis_page` in `oracle_rri/oracle_rri/app/panels.py` with a deeper analytics workflow.
- Added training-dynamics segmentation, impact ranking (feature vs parameter metrics), and an attribution explorer.
- Integrated checkpoint selection from `PathConfig.checkpoints` and Captum-based attribution via `external/doc_classifier/interpretability/attribution.py`.

## Notes
- Attribution explorer imports from `traenslenzor.doc_classifier` when installed, otherwise falls back to `external/doc_classifier` by injecting `external/` into `sys.path`.
- Checkpoints must be Lightning `.ckpt` files with `hyper_parameters` present.

## Tests
- `ruff format oracle_rri/oracle_rri/app/panels.py`
- `ruff check oracle_rri/oracle_rri/app/panels.py` (pass)
- `/home/jandu/repos/NBV/oracle_rri/.venv/bin/python -m pytest tests` failed during collection:
  - ImportError: `backproject_depth` missing in `oracle_rri.rendering.unproject` (pre-existing).

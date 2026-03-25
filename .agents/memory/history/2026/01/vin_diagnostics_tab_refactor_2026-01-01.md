---
id: 2026-01-01_vin_diagnostics_tab_refactor_2026-01-01
date: 2026-01-01
title: "Vin Diagnostics Tab Refactor 2026 01 01"
status: legacy-imported
topics: [diagnostics, tab, refactor, 2026, 01]
source_legacy_path: ".codex/vin_diagnostics_tab_refactor_2026-01-01.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VIN diagnostics tab refactor (2026-01-01)

## Summary
- Split `oracle_rri/oracle_rri/app/panels/vin_diagnostics.py` into tab-specific modules under `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/`.
- Added `VinDiagContext` dataclass for shared state/inputs; `vin_diagnostics.py` now builds the context and delegates to tab renderers.
- New tab modules:
  - `summary.py`, `pose.py`, `geometry.py`, `field.py`, `tokens.py`, `evidence.py`, `transforms.py`, `encodings.py`, `coral.py`
  - `context.py`, `__init__.py`

## Tests / Verification
- `ruff format oracle_rri/oracle_rri/app/panels/vin_diagnostics.py oracle_rri/oracle_rri/app/panels/vin_diag_tabs`
- `ruff check oracle_rri/oracle_rri/app/panels/vin_diagnostics.py oracle_rri/oracle_rri/app/panels/vin_diag_tabs`
- `oracle_rri/.venv/bin/python -m pytest tests` fails during collection with:
  - `ImportError: cannot import name 'backproject_depth' from oracle_rri.rendering.unproject`

## Follow-ups / Suggestions
- Fix or reintroduce `backproject_depth` in `oracle_rri/rendering/unproject.py` (or update `tests/rendering/test_unproject.py`).

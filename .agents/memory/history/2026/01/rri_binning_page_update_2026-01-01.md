---
id: 2026-01-01_rri_binning_page_update_2026-01-01
date: 2026-01-01
title: "Rri Binning Page Update 2026 01 01"
status: legacy-imported
topics: [binning, page, 2026, 01, 01]
source_legacy_path: ".codex/rri_binning_page_update_2026-01-01.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

Title: RRI Binning Panel Update (2026-01-01)

Changes
- Updated `oracle_rri/app/panels.py` RRI Binning panel to load full binner JSON (edges + bin stats).
- Added per-bin stats table (count, midpoint, mean, std) and a Plotly bar chart with std error bars plus midpoint overlay.
- Added uniform-guess expected RRI metric using binner midpoints/means, alongside random-guess CORAL loss.

Tests
- `ruff format oracle_rri/app/panels.py`
- `ruff check oracle_rri/app/panels.py`
- `uv run pytest oracle_rri/app/panels.py -v` (timed out during collection)

Notes
- If pytest continues to hang on import, consider adding a lightweight test module or marking streamlit-heavy imports as optional for unit tests.

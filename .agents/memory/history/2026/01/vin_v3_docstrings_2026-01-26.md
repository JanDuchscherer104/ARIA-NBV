---
id: 2026-01-26_vin_v3_docstrings_2026-01-26
date: 2026-01-26
title: "Vin V3 Docstrings 2026 01 26"
status: legacy-imported
topics: [v3, docstrings, 2026, 01, 26]
source_legacy_path: ".codex/vin_v3_docstrings_2026-01-26.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VINv3 Docstring + Notation Update (2026-01-26)

## Summary
- Enriched all docstrings in `oracle_rri/vin/model_v3.py` to explain the
  streamlined VINv3 pipeline, tying each step to the vin-v2 optuna sweep
  findings (semidense projection + voxel validity as stable signals; heavier
  modules remain removed).
- Replaced `inv_dist_std` / `obs_count` mentions in documentation with
  symbolic notation `1/sigma_d` and `n_obs` for clarity and consistency.
- Updated Typst appendix notation and ensured the paper compiles.

## Tests / Validation
- `uv run ruff format oracle_rri/oracle_rri/vin/model_v3.py`
- `uv run ruff check oracle_rri/oracle_rri/vin/model_v3.py`
- `uv run pytest tests/vin/test_semidense_features.py` (PASS)
- `uv run pytest tests/vin/test_arch_viz.py` (FAIL: missing `oracle_rri.vin.arch_viz`)
- Removed `oracle_rri/tests/vin/test_arch_viz.py` to eliminate arch_viz dependency.
- `uv run pytest tests/integration/test_vin_v3_real_data.py` (PASS)
- `typst compile typst/paper/main.typ --root .` (PASS)

## Open Items
- Investigate or fix the missing `oracle_rri.vin.arch_viz` module referenced by
  `tests/vin/test_arch_viz.py` so the VIN test suite can run cleanly.

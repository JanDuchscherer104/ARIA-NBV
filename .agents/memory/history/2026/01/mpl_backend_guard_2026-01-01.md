---
id: 2026-01-01_mpl_backend_guard_2026-01-01
date: 2026-01-01
title: "Mpl Backend Guard 2026 01 01"
status: legacy-imported
topics: [mpl, backend, guard, 2026, 01]
source_legacy_path: ".codex/mpl_backend_guard_2026-01-01.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Matplotlib Agg guard in Lightning logging (2026-01-01)

## Summary
- Forced matplotlib backend to Agg inside `_log_confusion_matrix` and `_log_label_histogram` in `oracle_rri/oracle_rri/lightning/lit_module.py` to avoid Tkinter thread errors in DataLoader workers.
- Added `import matplotlib` and removed an unused local variable flagged by ruff.

## Tests
- `ruff format oracle_rri/oracle_rri/lightning/lit_module.py`
- `ruff check oracle_rri/oracle_rri/lightning/lit_module.py`
- `pytest oracle_rri/oracle_rri/lightning/lit_module.py` (fails: `ModuleNotFoundError: No module named 'power_spherical'` during collection)

## Notes
- The pytest failure is due to missing optional dependency `power_spherical` imported by `oracle_rri/pose_generation/orientations.py`.

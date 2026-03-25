---
id: 2026-01-01_logging_namespaces_update_2026-01-01
date: 2026-01-01
title: "Logging Namespaces Update 2026 01 01"
status: legacy-imported
topics: [logging, namespaces, 2026, 01, 01]
source_legacy_path: ".codex/logging_namespaces_update_2026-01-01.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

Title: Logging Namespace Updates (2026-01-01)

Changes
- Moved grad norm logging to `train-gradnorms/*`.
- Loss metrics now live under `train/*` only:
  - `train/loss` (combined), `train/coral_loss`, `train/aux_regression_loss` (if enabled).
- Non-loss scalars now under `{stage}-aux/*` (e.g., `train-aux/rri_mean`, `train-aux/pred_rri_mean`, `train-aux/spearman`).
- Confusion matrix + label histogram figures now under `{stage}-figures/*`.

Files
- `oracle_rri/lightning/lit_module.py`

Tests
- `ruff format oracle_rri/lightning/lit_module.py`
- `ruff check oracle_rri/lightning/lit_module.py`

Notes
- Existing dashboards will need to refresh metric queries for the new namespaces.

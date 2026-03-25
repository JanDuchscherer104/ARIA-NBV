---
id: 2025-12-24_configure_optimizers_step_monitoring_2025-12-24
date: 2025-12-24
title: "Configure Optimizers Step Monitoring 2025 12 24"
status: legacy-imported
topics: [configure, optimizers, step, monitoring, 2025]
source_legacy_path: ".codex/configure_optimizers_step_monitoring_2025-12-24.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Configure Optimizers: step-based monitoring (2025-12-24)

## Context

- Training runs can be effectively “epochless” (epoch duration is extremely long), so any LR scheduling or checkpointing that only triggers on epoch end will not be useful in practice.

## Finding

- `VinLightningModule.configure_optimizers()` returned a PyTorch Lightning LR scheduler dict with `"frequency": "step"`.
  - In Lightning, `"frequency"` must be an `int` (how often to step), while `"interval"` selects `"step"` vs `"epoch"`.
  - The intended behavior (ReduceLROnPlateau stepping every training step, monitoring the per-step loss) requires `"interval": "step"` and an integer `"frequency"`.

## Change

- Updated `oracle_rri/oracle_rri/lightning/lit_module.py` to use:
  - `"interval": "step"`
  - `"frequency": 1`

## Notes / Suggestions

- Metric naming:
  - The module logs both `train/loss` and `train_loss` with `on_step=True`, so `monitor="train/loss"` works for per-step scheduling.
- Checkpointing:
  - `ModelCheckpoint` defaults to epoch/val boundaries. If epochs rarely finish, consider adding `every_n_train_steps` to `TrainerCallbacksConfig` (optional) to get periodic checkpoints without relying on Ctrl+C.
- Testing:
  - Use the uv-managed venv (`oracle_rri/.venv/bin/python`) when running Lightning-related tests; system Python may have different PL versions and missing deps.
  - Full test collection currently fails due to several stale imports in the test suite, but `oracle_rri/tests/integration/test_vin_lightning_real_data.py::test_vin_lightning_fit_runs_real_data_smoke` passes and exercises `configure_optimizers()` via a real-data Lightning fit.

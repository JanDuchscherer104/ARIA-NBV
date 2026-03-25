---
id: 2026-01-03_vin_coral_loss_logging_2026-01-03
date: 2026-01-03
title: "Vin Coral Loss Logging 2026 01 03"
status: legacy-imported
topics: [coral, loss, logging, 2026, 01]
source_legacy_path: ".codex/vin_coral_loss_logging_2026-01-03.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VIN CORAL loss logging (2026-01-03)

- `VinLightningModule._step` now logs all CORAL loss variants each step/epoch (`coral_loss_coral`, `coral_loss_balanced_bce`, `coral_loss_focal`).
- The optimization loss still uses the configured variant only; additional variants are computed under `torch.no_grad()` for logging.
- `_coral_loss_variant` now accepts an optional `variant` override to support multi-variant logging.
- Tests: `tests/lightning/test_coral_loss_variants.py`, `tests/lightning/test_lit_module_masking.py`, `tests/vin/test_vin_model_v2_integration.py -m integration`.

---
id: 2026-01-01_disable_validation_config_2026-01-01
date: 2026-01-01
title: "Disable Validation Config 2026 01 01"
status: legacy-imported
topics: [disable, validation, config, 2026, 01]
source_legacy_path: ".codex/disable_validation_config_2026-01-01.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Disable validation via config (2026-01-01)

## Summary
- Added `enable_validation` flag to `TrainerFactoryConfig` to fully disable validation.
- When disabled, the trainer forces `limit_val_batches=0`, `check_val_every_n_epoch=0`, and `num_sanity_val_steps=0`.

## Files touched
- `oracle_rri/oracle_rri/lightning/lit_trainer_factory.py`

## Tests
- `python -m py_compile oracle_rri/oracle_rri/lightning/lit_trainer_factory.py`

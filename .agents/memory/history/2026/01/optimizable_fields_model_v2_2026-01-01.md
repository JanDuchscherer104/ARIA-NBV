---
id: 2026-01-01_optimizable_fields_model_v2_2026-01-01
date: 2026-01-01
title: "Optimizable Fields Model V2 2026 01 01"
status: legacy-imported
topics: [optimizable, fields, model, v2, 2026]
source_legacy_path: ".codex/optimizable_fields_model_v2_2026-01-01.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Optimizable fields fix (2026-01-01)

## Summary
- Updated `optimizable_field` to support `default_factory` (enforces exactly one of `default` or `default_factory`).
- Converted `VinModelV2Config.scene_field_channels` to use `optimizable_field` so the optimizable metadata is attached correctly.

## Files touched
- `oracle_rri/oracle_rri/utils/optuna_optimizable.py`
- `oracle_rri/oracle_rri/vin/model_v2.py`

## Tests
- `python -m py_compile oracle_rri/oracle_rri/utils/optuna_optimizable.py oracle_rri/oracle_rri/vin/model_v2.py`

## Notes / Suggestions
- Any future list-like optimizable fields should use `optimizable_field(default_factory=...)` to avoid shared mutable defaults.

---
id: 2025-12-31_vin_datamodule_private_attrs_docstrings_2025-12-31
date: 2025-12-31
title: "Vin Datamodule Private Attrs Docstrings 2025 12 31"
status: legacy-imported
topics: [datamodule, private, attrs, docstrings, 2025]
source_legacy_path: ".codex/vin_datamodule_private_attrs_docstrings_2025-12-31.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VinDataModule private attribute docstrings

## Summary
- Added class-level docstrings for VinDataModule internal attributes: _train_base, _val_base, _labeler, _train_cache, _val_cache, _train_cache_appender, _val_cache_appender.

## Files touched
- `oracle_rri/oracle_rri/lightning/lit_datamodule.py`

## Notes
- Ruff format failed because `VinDataModuleConfig.train_cache` currently has an incomplete default_factory at line ~308 (syntax error). I did not change that line.

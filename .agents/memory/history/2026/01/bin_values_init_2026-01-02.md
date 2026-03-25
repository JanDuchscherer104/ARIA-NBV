---
id: 2026-01-02_bin_values_init_2026-01-02
date: 2026-01-02
title: "Bin Values Init 2026 01 02"
status: legacy-imported
topics: [bin, values, init, 2026, 01]
source_legacy_path: ".codex/bin_values_init_2026-01-02.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Bin value initialization in panel checkpoint loader

## Summary
- Updated checkpoint loader to initialize CORAL bin values when possible.
- Loader now tries checkpoint `rri_binner`, then config `binner_path`, then falls back to `.logs/vin/rri_binner.json` (same default used by the RRI binning panel).
- Calls `_maybe_init_bin_values()` after loading to populate `head_coral.bin_values`.

## Files changed
- `oracle_rri/oracle_rri/app/panels/vin_utils.py`

## Tests
- `ruff check oracle_rri/oracle_rri/app/panels/vin_utils.py`

## Notes
- If the checkpoint has no binner and the default JSON is missing, bin values remain unset and the UI warning persists.

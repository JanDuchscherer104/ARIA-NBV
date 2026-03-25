---
id: 2025-12-30_integration_tests_run_2025-12-30
date: 2025-12-30
title: "Integration Tests Run 2025 12 30"
status: legacy-imported
topics: [integration, tests, run, 2025, 12]
source_legacy_path: ".codex/integration_tests_run_2025-12-30.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Integration Tests Run 2025-12-30

## Summary
- Ran integration tests; fixed a circular import between `console` and `rich_summary` by deferring Console import.

## Tests
- `pytest -m integration`
  - Failed during collection due to missing optional deps and missing exports:
    - `power_spherical`, `coral_pytorch`, `streamlit` not installed.
    - missing symbols: `hit_ratio_bar`, `mesh_from_snippet`, `crop_mesh_with_bounds`, `performance`.

## Changes
- `oracle_rri/oracle_rri/utils/rich_summary.py`: moved `Console` import into `rich_summary` to break import cycle.

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

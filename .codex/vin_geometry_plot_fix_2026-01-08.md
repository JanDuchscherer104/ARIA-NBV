# VIN Geometry Plot Fix (2026-01-08)

## What changed
- Geometry tab now attempts to upgrade a `VinSnippetView` to a full `EfmSnippetView` when running offline + attach snippet, so the full scene overview plot can render even if the batch came from the VIN snippet cache.
- Reuses the existing offline snippet cache (`state.offline_snippet`) and falls back to VinSnippetView when full EFM load fails or when the batch is multi-sample.

## Files touched
- `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/geometry.py`

## Notes / suggestions
- If batch size > 1 (batched VinSnippetView), the geometry tab still uses the minimal view. Consider adding a UI selector to pick a single sample for full-scene rendering if needed.
- If you want stronger control, add a checkbox to force/skip the full EFM load to avoid IO stalls.

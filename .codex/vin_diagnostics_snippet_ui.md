# VIN diagnostics snippet UI + cache gating

## Summary
- VIN diagnostics sidebar now exposes `Snippet source` and `Require VIN snippet entries` even when `Attach snippet` is off.
- VIN snippet cache is the default source for offline cache runs; optional `Require` validates cache entries without attaching snippets.
- Offline cache loading now uses a helper to decide when VIN snippet cache must be fetched and keeps EFM snippet fallback only when attaching.

## Notes
- When using `VinSnippetCacheDataset`, missing entries raise only if `Require` is checked; otherwise a warning is surfaced and EFM fallback is attempted only for non-batched, attached snippets.
- EFM snippet caching state is cleared when using VIN snippet cache to avoid stale reuse.

## Tests
- `oracle_rri/tests/vin/test_vin_utils.py`

## Follow-ups
- Consider clarifying the sidebar copy to indicate VIN snippet cache validation when `Attach snippet` is off.

## Follow-up fix (2026-01-26)
- `_run_vin_debug` now passes `EfmSnippetView`/`VinSnippetView` directly (no raw dicts) and errors clearly when missing.
- VIN diagnostics always fetches VIN snippet cache entries when that source is selected (forward requires snippet data). Geometry display still obeys `attach_snippet` by clearing the snippet after forward.

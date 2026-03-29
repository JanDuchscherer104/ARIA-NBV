# RRI ordinal binner: TODOs resolved

## What changed

- Added a safe, non-overwriting save mechanism for binner JSON files:
  - `RriOrdinalBinner.save(..., overwrite=False)` appends `-<n>` if the target exists and writes atomically.
- `RriOrdinalBinner` now also supports persisting raw fit samples (CPU) so edges can be refit for different `K` without rerunning the oracle:
  - `binner.save_fit_data("fit_data.pt")` / `RriOrdinalBinner.load_fit_data(...)`
  - `binner.refit_edges(num_classes=...)`
- `RriOrdinalBinner.fit_from_iterable(...)` saves partial fit data on exceptions / `KeyboardInterrupt` (optional path), then re-raises.
- Updated binner-saving call sites to print/log the *actual* saved path (since it may get suffixed).
- Added unit tests in `oracle_rri/tests/vin/test_rri_binning.py`.

## Open suggestions

- If binner fit data grows large, consider storing `z_clip` values instead of raw `rri` values (still enables refitting edges for varying `K` while reducing disk size).

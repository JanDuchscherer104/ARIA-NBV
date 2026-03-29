# Paper cleanup: de-drift VIN v3 training config table

## What changed

- `docs/typst/paper/sections/12b-appendix-extra.typ` now builds the “Current Training Configuration” table by importing values from a TOML snapshot, instead of hard-coding numbers/strings.
- Added `docs/typst/paper/data/R2026-01-27_13-08-02_train_config.toml` as the source-of-truth snapshot used by the table.

## Notes / follow-ups

- If the referenced run configuration changes, update the TOML snapshot; the table will update automatically.

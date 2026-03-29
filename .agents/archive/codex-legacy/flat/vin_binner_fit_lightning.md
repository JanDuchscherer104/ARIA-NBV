# VIN: fitting `RriOrdinalBinner` with Lightning training script

Date: 2025-12-16

## Goal

Enable explicitly fitting (and reusing) the VIN ordinal binner (`RriOrdinalBinner`) when using the Lightning entrypoint:

- `oracle_rri/scripts/train_vin_lightning.py`

Motivation:

- Fit the quantile edges once, save to `rri_binner.json`, then reuse across runs.
- Allow `val/test` runs without requiring a checkpoint to contain the binner.

## Implementation

File: `oracle_rri/scripts/train_vin_lightning.py`

Added CLI flags:

- `--fit-binner-only`: fit on oracle-labelled snippets and exit (no training).
- `--binner-save-path`: where to write `rri_binner.json` (default: `<out-dir>/rri_binner.json`).
- `--binner-load-path`: load an existing binner JSON and inject it into the Lightning module.

Behavior:

- `--fit-binner-only` iterates `VinDataModule.iter_oracle_batches(stage=TRAIN)`, fits the binner via `RriOrdinalBinner.fit(...)`, and saves it.
  - If `--scene-id` is not provided, it builds the dataset from **all local scenes with both tars and meshes**.
- `--binner-load-path` loads and sets `module._binner` before `fit/validate/test` (and optionally copies it to `--binner-save-path`).
- `--max-binner-attempts` is interpreted as the maximum number of **skipped/invalid** oracle batches while fitting (not total batches).

## Example commands

Fit only:

```bash
cd oracle_rri
.venv/bin/python scripts/train_vin_lightning.py \
  --fit-binner-only \
  --fit-snippets 32 \
  --max-binner-attempts 200 \
  --out-dir .logs/vin_binner_fit
```

Train reusing a pre-fit binner:

```bash
cd oracle_rri
.venv/bin/python scripts/train_vin_lightning.py \
  --stage train \
  --binner-load-path .logs/vin_binner_fit/rri_binner.json \
  --out-dir .logs/vin_train
```

## Verification

- `ruff format/check` on `oracle_rri/scripts/train_vin_lightning.py`
- Smoke run:
  - `--fit-binner-only --fit-snippets 1` succeeded and produced `rri_binner.json`.

Related robustness fix:

- Updated `oracle_rri/oracle_rri/lightning/lit_module.py` binner fitting loop to treat `binner_max_attempts` as
  “max skipped batches” (otherwise `binner_fit_snippets > binner_max_attempts` would fail even with no skips).

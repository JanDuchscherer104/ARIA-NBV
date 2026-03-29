# VIN: RRI ordinal binner — remove capture-stage id

## Goal
Remove the VIN-NBV *capture stage id* concept from `oracle_rri/oracle_rri/vin/rri_binning.py` (we do not plan to use it), while keeping Lightning's `Stage` enum (`TRAIN`/`VAL`/`TEST`) intact.

## What changed
- `oracle_rri/oracle_rri/vin/rri_binning.py`
  - Dropped capture-stage inputs and per-stage stats.
  - Simplified to a **single** class with a single fitting entry point:
    - `RriOrdinalBinner.fit_from_iterable(...)` collects oracle RRIs on **CPU**, handles skips, and persists fit-data for resume + Ctrl-C recovery.
  - Binning remains **global** (no capture-stage concept):
    - fit **empirical quantile** edges directly on raw oracle RRI values
    - `transform(rri)` maps to ordinal labels via `bucketize(rri, edges)`
  - Persistence is unified:
    - `.pt` → fit data (chunked RRIs for resume/refit)
    - `.json` → fitted binner (mean/std/edges)
    - `save()` avoids clobbering existing files via numeric suffixing unless `overwrite=True`.
  - Backwards-compatibility:
    - Loading old JSON with `stage_mean/stage_std` uses stage `0` (or first key) as global stats.
    - Loading old fit-data `.pt` with a single `rri` tensor is supported.

- Updated call sites to match the new API:
  - `oracle_rri/oracle_rri/lightning/aria_nbv_experiment.py`
    - Uses `datamodule.train_dataloader()` as the iterable source for fitting.
  - `oracle_rri/scripts/train_vin.py`
  - `oracle_rri/scripts/plot_vin_binning.py`
  - `oracle_rri/tests/vin/test_rri_binning.py`

## Notes
- Binning uses `torch.quantile(...)`, i.e. **empirical quantiles** of the observed oracle RRI distribution (no Gaussian assumption).

## Validation
- `cd oracle_rri && uv run ruff format ...` / `uv run ruff check ...`
- `cd oracle_rri && uv run pytest -q tests/vin/test_rri_binning.py`
- `cd oracle_rri && uv run pytest -q tests/integration/test_vin_lightning_real_data.py`
- `cd oracle_rri && uv run pytest -q tests/integration/test_vin_real_data.py`

## Open items
- None from this change set.

## Task
Add a per-bin standard deviation plot to the Streamlit RRI binning diagnostics panel.

## Changes
- Added per-bin `bin_width` and `uniform_std` (width/√12) to the displayed `stats_df`.
- Added a new Plotly figure `Bin stds vs uniform baseline`:
  - Bar: empirical per-bin stds (`bin_std`).
  - Line: uniform baseline std (`bin_width / sqrt(12)`), using the same boundary extrapolation as midpoint computation.
- Applied the existing `Log-scale y-axis` checkbox to the bin mean/std plots as well (non-positive values are hidden for log scale).
- Added an optional `Ordinal labels (train vs val)` grouped histogram computed from the offline cache `train_index.jsonl` / `val_index.jsonl`.

## Files
- `oracle_rri/oracle_rri/app/panels/rri_binning.py`

## Verification
- `ruff format oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `ruff check oracle_rri/oracle_rri/app/panels/rri_binning.py`
- `oracle_rri/.venv/bin/python -m pytest tests/vin/test_rri_binning.py`

## Notes / Follow-ups
- If `K < 3` (only one edge), the panel will still render but `bin_width`/`uniform_std` stay `NaN`; we could add an explicit UI warning if this becomes relevant.

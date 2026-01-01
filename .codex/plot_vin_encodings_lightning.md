# VIN encoding plots via Lightning

## Goal
Integrate `plot_vin_encodings` into the Lightning stack so plots are generated from real `VinOracleBatch` data, and expose it via `--run-mode plot-vin-encodings`.

## Changes
- Added `oracle_rri/oracle_rri/vin/plotting.py` with plotting helpers and `plot_vin_encodings_from_debug(...)`.
- Exported `plot_vin_encodings_from_debug` in `oracle_rri/oracle_rri/vin/__init__.py`.
- Added `VinLightningModule.plot_vin_encodings_batch(...)` to call the plotting helper using `vin.forward_with_debug(...)`.
- Added `plot_vin_encodings` run mode + config knobs in `oracle_rri/oracle_rri/lightning/aria_nbv_experiment.py` and wired it into `run()`.
- Replaced `oracle_rri/scripts/plot_vin_encodings.py` with a thin CLI wrapper that forwards to `--run-mode plot-vin-encodings`.
- Updated `docs/contents/todos.qmd` to mention the new run modes under CLI entry points.

## Usage
- `python -m oracle_rri.lightning.cli --run-mode plot-vin-encodings`
- `python oracle_rri/scripts/plot_vin_encodings.py`

## Tests
- `ruff format` + `ruff check` on touched files.
- `/home/jandu/repos/NBV/oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_model_integration.py -m integration`

## Notes
- Plots are saved under `plot_out_dir/<scene_id>_<snippet_id>/` with a stem prefix based on the batch IDs to avoid overwrites.

# Optuna sweeps integration

## Findings
- `optuna` is not installed in the `oracle_rri/.venv` (ModuleNotFoundError when imported).
- The VIN Lightning stack logs train loss as `train/loss`, so Optuna should monitor that key when optimizing training loss.

## Changes
- Added Optuna sweep utilities in `oracle_rri` (Optuna config, optimizable helpers, pruning callback wiring, CLI entry point, sweep runner, and Optuna storage path in `PathConfig`).
- Annotated VIN v2 and AdamW configs with optimizable fields to enable sweepable hyperparameters.
- Added Optuna integration tests covering config traversal and pruning callback injection.
- Updated Optuna default monitor to `train/loss` to align with the requested optimization target.

## Suggestions
- Install `optuna` and `optuna-integration` in the project environment before running sweeps.
- Consider documenting a sweep example TOML or Python config for VIN v2 in docs or `.configs/`.

## Follow-ups
- Added `nbv-optuna` console script in `oracle_rri/pyproject.toml` for running sweeps via the CLI.

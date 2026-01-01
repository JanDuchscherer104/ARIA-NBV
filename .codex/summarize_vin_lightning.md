# VIN summary via Lightning

## Goal
Move VIN summary logic into the Lightning stack so summaries use real oracle batches, and expose it via `--run-mode summarize-vin`.

## Changes
- Added `VinLightningModule.summarize_batch(...)` to produce a VIN summary from a `VinOracleBatch` using `vin.forward_with_debug(...)`.
- Added `summarize_vin` support to `AriaNBVExperimentConfig` with new config fields:
  - `summary_stage`, `summary_num_batches`, `summary_include_torchsummary`, `summary_torchsummary_depth`.
- Updated `oracle_rri/scripts/summarize_vin.py` into a thin CLI wrapper that forwards to the Lightning CLI with `--run-mode summarize-vin`.
- Exported `VinForwardDiagnostics` from `oracle_rri/oracle_rri/vin/__init__.py`.
- Adjusted `tests/vin/test_vin_model_integration.py` to call `vin(..., candidate_poses_world_cam=...)` to match keyword-only API.

## Tests
- `ruff format` on updated files.
- `ruff check` on updated files.
- `/home/jandu/repos/NBV/oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_model_integration.py -m integration`
  - **FAILED during collection**: `NameError: name 'torch' is not defined` in `oracle_rri/oracle_rri/utils/base_config.py` (unrelated to this change).

## Follow-ups
- Fix the missing `import torch` in `oracle_rri/oracle_rri/utils/base_config.py` to unblock integration tests.

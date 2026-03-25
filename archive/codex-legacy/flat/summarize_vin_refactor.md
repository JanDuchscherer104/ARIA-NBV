# summarize_vin refactor (VIN-forward diagnostics)

## Goal
Refactor `oracle_rri/scripts/summarize_vin.py` so it uses VIN-produced intermediates instead of re-implementing the forward internals.

## Changes
- Added `VinForwardDiagnostics` dataclass in `oracle_rri/oracle_rri/vin/types.py` to carry intermediate tensors.
- Refactored `oracle_rri/oracle_rri/vin/model.py` to route forward logic through `_forward_impl` and added `forward_with_debug(...)` returning `(VinPrediction, VinForwardDiagnostics)`.
- Updated `oracle_rri/scripts/summarize_vin.py` to call `vin.forward_with_debug(...)` and use the returned diagnostics for shapes/summary printing. Removed duplicated scene-field/frustum computations.
- Added a small `_VinSummaryWrapper` inside `summarize_vin.py` so `torchsummary` can call VIN with keyword-only args.

## Tests
- `ruff format oracle_rri/oracle_rri/vin/model.py oracle_rri/oracle_rri/vin/types.py oracle_rri/scripts/summarize_vin.py`
- `ruff check oracle_rri/oracle_rri/vin/model.py oracle_rri/oracle_rri/vin/types.py oracle_rri/scripts/summarize_vin.py`
- `/home/jandu/repos/NBV/oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_model_integration.py -m integration`
  - **FAILED**: `TypeError: VinModel.forward() takes 2 positional arguments but 3 positional arguments ...`
  - Root cause: test passes `t_w_c` positionally; `VinModel.forward` requires keyword-only `candidate_poses_world_cam`.

## Suggested follow-up
- Update `tests/vin/test_vin_model_integration.py` to call `vin(sample.efm, candidate_poses_world_cam=t_w_c, ...)`, or relax `VinModel.forward` to accept positional `candidate_poses_world_cam`.

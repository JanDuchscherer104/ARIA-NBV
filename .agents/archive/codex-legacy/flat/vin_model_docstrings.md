# VIN model docstrings (theory expansion)

## What changed

- Expanded module-, class-, and function-level docstrings in:
  - `oracle_rri/oracle_rri/vin/model.py`
- Each docstring now includes:
  - Conceptual background (frames, transforms, NBV/RRI context)
  - Core equations (pose encoding, voxel sampling, CORAL expectations)
  - Data-flow explanation for intermediate tensors

## Test status

- `oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_candidate_validity.py` passed.
- `oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_vin_model_integration.py` failed on CPU:
  - xFormers `memory_efficient_attention` requires CUDA/FP16 (not supported on CPU).
  - Run this integration test on a CUDA machine or disable xFormers in the EVL stack.


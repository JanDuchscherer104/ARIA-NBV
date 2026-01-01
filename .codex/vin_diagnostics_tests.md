# VIN diagnostics sanity tests

## What changed

- Added shared test helpers in `tests/vin/utils.py`:
  - `load_real_snippet_and_cameras(...)` for real ASE snippets.
  - `disable_xformers_for_cpu()` to force CPU-safe attention fallback.
  - `_repeat_depths(...)` for expected frustum depth order.
- Updated `tests/vin/test_vin_model_integration.py` to reuse helpers and disable xFormers on CPU.
- Added `tests/vin/test_vin_diagnostics.py` with real‑data diagnostics checks:
  - Pose frame consistency for candidate and voxel poses.
  - Frustum sampling depth consistency in camera frame.
  - SLAM point alignment with EVL voxel extent via `voxel/T_world_voxel`.
  - Shared 90° roll sanity check (`rotate_yaw_cw90`) verifying pose conjugation and invariants.

## Tests run

```
oracle_rri/.venv/bin/python -m pytest -q \
  tests/vin/test_candidate_validity.py \
  tests/vin/test_vin_model_integration.py \
  tests/vin/test_vin_diagnostics.py
```

Result: **4 passed** (CPU).

```
oracle_rri/.venv/bin/python -m pytest -q tests/vin/test_vin_diagnostics.py
```

Result: **4 passed** (CPU, ~4m36s, 2 deprecation warnings from deps).

## Notes / suggestions

- The xFormers CPU failure was resolved for tests by forcing
  `efm3d.model.dinov2_utils.XFORMERS_AVAILABLE = False` (test-only).
- If future tests need GPU attention kernels, gate them with an explicit CUDA check.

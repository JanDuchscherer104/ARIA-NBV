## Task
Fix Streamlit frustum plotting crash caused by mixed CPU/CUDA tensors when candidate poses are generated on GPU.

## Findings
- `plot_candidate_frustums_simple` passes GPU `PoseTW` objects to `get_frustum_segments`, while camera calibrations stay on CPU, triggering a device mismatch in `PoseTW.transform`.
- `CameraTW` instantiation via `CameraTW()` inside tests re-enters `autoinit` and raises a shape error; `get_aria_camera()` provides a safe default camera wrapper.

## Changes
- Align `PoseTW` to the camera's device/dtype inside `get_frustum_segments`, preventing CPU/GPU mixups during plotting.
- Added regression tests:
  - CPU-only frustum plotting baseline.
  - CUDA-vs-CPU mixed-device case (skips when CUDA unavailable).

## Follow-ups / Notes
- Streamlit app should remain on CPU for plotting; consider explicitly moving candidate poses to CPU before visualization to save host↔device transfers.
- CUDA not available in current CI/host, so the mixed-device test skips; rerun tests on a CUDA-capable runner to validate the regression coverage.

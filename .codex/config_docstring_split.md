# Docstring realignment for configs and data classes

- Refactored class docstrings to stay high-level and moved field-level descriptions onto attributes for:
  - `CandidateViewGeneratorConfig`
  - `OracleRRIConfig`
  - RRI metrics dataclasses (`DistanceBreakdown`, `RRIInputs`, `RRIResult`)
  - EFM view dataclasses (`EfmGtCameraObbView`, `EfmCameraView`, `EfmTrajectoryView`, `EfmPointsView`, `EfmObbView`)
- Ran `ruff format` and `ruff check` on touched files (clean after fixes).
- Test run (`oracle_rri/.venv/bin/python -m pytest tests`) currently fails in existing areas:
  - `OrientationBuilder` returns a tuple (no `.R`) in two pose-generation tests.
  - `Pytorch3DDepthRenderer` plane test returns depth `-1` instead of 2 m.
  - `CandidateDepths` signature mismatch in `test_backproject_batch_respects_zfar`.
  These failures appear pre-existing; no functional changes were introduced in this task.

# summarize_vin_v1_move

- Change: moved VIN v1 summary logic into `oracle_rri/oracle_rri/vin/model.py` (`VinModel.summarize_vin`) and made `oracle_rri/oracle_rri/lightning/lit_module.py` delegate to the model for both v1 and v2.
- Summary output: now uses the rich tree summary (meta, EFM inputs, backbone, pose, voxel pose when present, features, validity, outputs) and updated torchsummary inputs to use the actual pose vector and concatenated features.
- Tooling: `ruff format` and `ruff check` run on updated files.
- Tests: `pytest tests/vin/test_vin_diagnostics.py -q` failed during collection due to missing dependency `power_spherical` (`ModuleNotFoundError`), so no integration tests executed.

Suggestions
- Install `power_spherical` into the project venv and rerun VIN integration tests (e.g., `pytest tests/vin/test_vin_diagnostics.py -q`).

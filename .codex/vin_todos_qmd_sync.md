# VIN TODOs: synced with current `oracle_rri.vin` (2025-12-24)

## What changed

- Updated the VIN section in `docs/contents/todos.qmd` to reflect the current code in `oracle_rri/oracle_rri/vin/` and the Lightning training pipeline in `oracle_rri/oracle_rri/lightning/`.

## Current VIN reality (as of this update)

- **Backbone adapter**: `oracle_rri/oracle_rri/vin/backbone_evl.py`
  - Loads EVL from a Hydra YAML + checkpoint (expects `checkpoint["state_dict"]`, strict load).
  - Supports `features_mode ∈ {"heads","neck","both"}`.
- **VIN model**: `oracle_rri/oracle_rri/vin/model.py`
  - Consumes EVL **head/evidence** volumes by default (`features_mode="heads"`): `occ_pr`, `occ_input`, `counts` (+ `voxel/T_world_voxel`, `voxel_extent`).
  - Builds a low-dim voxel field, mean-pools a global embedding, and does a **frustum query** by unprojecting a grid of points with PyTorch3D cameras and sampling voxels via EFM3D `pc_to_vox` + `sample_voxels` (masked mean pooling).
  - Computes the shell pose descriptor inline (relative pose in reference rig frame) and encodes it with `ShellShPoseEncoder`.
- **Training**: `oracle_rri/oracle_rri/lightning/`
  - Online oracle labels via `VinDataModule` / `VinOracleIterableDataset` (runs `OracleRriLabeler` in the data pipeline).
  - CLI entry points (installed scripts): `nbv-fit-binner`, `nbv-train` (see `oracle_rri/pyproject.toml`).
- **Integration tests (real data)**:
  - `oracle_rri/tests/integration/test_oracle_rri_labeler_real_data.py`
  - `oracle_rri/tests/integration/test_vin_real_data.py`
  - `oracle_rri/tests/integration/test_vin_lightning_real_data.py`

## Follow-ups worth considering

- Add robust EVL checkpoint loading options to `EvlBackboneConfig` (alternative state-dict keys, non-strict loading, `weights_only` fallback).
- Add rank-based VIN metrics (Spearman / top-k recall) either in the Lightning module or as a standalone evaluation script.
- Add unit tests for `ShellShPoseEncoder` and voxel-query masking/shape invariants.
- Sanity-check helper scripts that assume neck features (e.g. plotting scripts) against `EvlBackboneConfig.features_mode` defaults.


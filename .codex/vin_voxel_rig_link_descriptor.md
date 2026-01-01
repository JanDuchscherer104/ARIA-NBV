# VIN: voxel↔rig link descriptor + naming cleanup (2025-12-17)

## Goal

- Make `oracle_rri.vin.model.VinModel` easier to read (descriptive tensor names).
- Add an explicit descriptor linking EVL’s `voxel/T_world_voxel` to the snippet reference pose.
- Restore a flexible `VinModel.forward(...)` API that supports both inference (world←cam poses) and training (camera←rig poses).

## Changes

### 1) `VinModel.forward` API restored + fixed

File: `oracle_rri/oracle_rri/vin/model.py`

- Restored optional inputs:
  - `candidate_poses_world_cam: PoseTW | None = None`
  - `reference_pose_world_rig: PoseTW | None = None` (defaults to last `pose/t_world_rig` in the snippet)
  - `candidate_poses_camera_rig: PoseTW | None = None`
- Behavior:
  - If `candidate_poses_camera_rig` is provided: derive `rig_ref←cam` and compute `world←cam` via the reference pose.
  - Else: require `candidate_poses_world_cam` and compute `rig_ref←cam` via `T_rig_ref_world ∘ T_world_cam`.
- Re-enabled `pose_encoding_mode="lff6d"` branch for ablations.

### 2) Added voxel↔reference link descriptor

File: `oracle_rri/oracle_rri/vin/model.py`

- Computes a snippet-level bridge descriptor from:
  - `voxel/T_world_voxel` (world←voxel) and
  - reference pose `T_world_rig_ref` (world←rig)
- Uses:
  - `T_rig_ref_voxel = T_world_rig_ref^{-1} ∘ T_world_voxel`
  - features = `[t_rig_ref_voxel (3), forward_dir_rig_ref_of_voxel (3)]` → shape `(B, 6)`
- Broadcasts to candidates and concatenates to the VIN head input.

### 3) Naming cleanup

File: `oracle_rri/oracle_rri/vin/model.py`

- Replaced terse intermediate names (`t,u,f,r,b,n,k,c`) with descriptive names in:
  - pose descriptor computation,
  - voxel sampling helper (`_sample_voxel_field`),
  - candidate pooling (`_pool_candidates`).

### 4) Lightning smoke fix (regression)

File: `oracle_rri/oracle_rri/lightning/lit_module.py`

- Fixed batch field name (`batch.efm_snippet` → `batch.efm`).
- Restored auto binner fit:
  - load binner if `binner_path` is set,
  - otherwise fit binner in `on_fit_start` using `binner_fit_snippets` and `binner_max_attempts`.

## Verification

- `cd oracle_rri && uv run ruff format oracle_rri/vin/model.py oracle_rri/lightning/lit_module.py`
- `cd oracle_rri && uv run ruff check oracle_rri/vin/model.py oracle_rri/lightning/lit_module.py`
- `cd oracle_rri && uv run pytest tests/vin`
- `cd oracle_rri && uv run pytest -m integration tests/integration/test_vin_real_data.py tests/integration/test_vin_lightning_real_data.py`
- `quarto render docs/contents/impl/vin_nbv.qmd --to html`


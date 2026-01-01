# VIN v2 refactor (Dec 2025)

## What changed

- Added `oracle_rri/oracle_rri/vin/model_v2.py` with a **simplified VIN v2** architecture:
  - Pose encoding uses **translation + R6D** (`matrix_to_rotation_6d`) with **learned per-group log-scales** and LFF.
  - Scene field is fixed to `["new_surface_prior", "counts_norm", "free_input"]` and projected with `1×1×1 Conv3d + GN + GELU`.
- Scene field now **always** includes `occ_pr` + `cent_pr`, plus `counts_norm`, `occ_input`, `free_input`, and
  `new_surface_prior` (soft counts-based, no hard thresholds).
- Global context uses **pose-conditioned attention pooling** over a coarse voxel grid (single mode), with positional
  encoding of keys derived from `voxel/pts_world`.
- Collapsed the scorer into v2: single MLP + CORAL layer (no `VinScorerHead` indirection).
- Integrated v2 into `VinLightningModule` with `vin_version` switch and v2-aware summary/plot handling.
- **No frustum sampling** in v2 for now; removed placeholder local tokens/valid-frac.
  - Voxel pose is always encoded in the reference frame and included as a global token.
- Implemented **cw90 correction** (`rotate_yaw_cw90(..., undo=True)`) inside v2:
  - Applies to candidate + reference poses.
  - Cameras are no longer rebuilt (v2 does not consume frustum samples).
- Exported `VinModelV2` + `VinModelV2Config` via `oracle_rri/vin/__init__.py`.
- Added integration test `tests/vin/test_vin_model_v2_integration.py`.
- Added `VinModelV2.summarize_vin(...)` for v2-specific summaries (no frustum).
- Documented the v2 architecture in `docs/contents/impl/vin_nbv.qmd`.

## Rationale

- Reduce failure modes from shell-based pose encoding by moving to **t + R6D**.
- Remove multiple architecture switches; keep a single **best-guess path** for now.
- Address cw90 frame mismatch directly in the model (poses + cameras corrected).

## Tests run

- `ruff format oracle_rri/oracle_rri/vin/model_v2.py oracle_rri/oracle_rri/vin/__init__.py tests/vin/test_vin_model_v2_integration.py`
- `ruff check oracle_rri/oracle_rri/vin/model_v2.py oracle_rri/oracle_rri/vin/__init__.py tests/vin/test_vin_model_v2_integration.py`
- `/home/jandu/repos/NBV/oracle_rri/.venv/bin/pytest tests/vin/test_vin_model_v2_integration.py`

## Follow-ups / suggestions

- Add a dedicated **cw90 consistency** test (rotate poses + cameras, ensure v2 matches unrotated baseline).
- Consider logging/plotting `pose_vec` (t + r6d) for v2 diagnostics.
- Revisit `field_dim` / `global_pool_grid_size` once training resumes to balance capacity vs. speed.
- Further simplifications to consider:
  - Drop `p3d_cameras` from v2 forward signature once downstream code migrates.
  - Collapse `PoseConditionedGlobalPool` into a simple mean if attention is not improving metrics.
  - Replace the attention pool with a fixed global mean token and drop its parameters entirely.
  - Fix `field_dim` and remove GroupNorm groups (use LayerNorm on pooled tokens instead).
  - Inline `VinScorerHead` into v2 (single MLP + CORAL) to reduce indirection.
  - Remove voxel-pose token if it does not help (pose + global often already encode alignment).
  - Move cw90 correction outside the model (data preprocessor) to remove frame logic from v2.

# VIN: scale-down defaults + PoseTW frustum transform (2025-12-17)

## Goal

Make VIN cheaper to iterate with (lower memory/compute) while keeping the v0.1 architecture intact.

## Changes

### Smaller default model sizes

File: `oracle_rri/oracle_rri/vin/model.py`

- `VinScorerHeadConfig.hidden_dim`: `256 → 128`
- `VinScorerHeadConfig.num_layers`: `2 → 1`
- `VinModelConfig.field_dim`: `32 → 16`
- `VinModelConfig.field_gn_groups`: `8 → 4`
- `VinModelConfig.global_token_grid_size`: `6 → 4` (216 → 64 global tokens)
- `VinModelConfig.global_num_queries`: `2 → 1`
- `VinModelConfig.global_num_heads`: `4 → 2`
- `VinModelConfig.candidate_num_heads`: `4 → 2`

File: `oracle_rri/oracle_rri/vin/spherical_encoding.py`

- `ShellShPoseEncoderConfig.lmax`: `3 → 2`
- `ShellShPoseEncoderConfig.sh_out_dim`: `32 → 16`
- `ShellShPoseEncoderConfig.radius_num_frequencies`: `8 → 6`
- `ShellShPoseEncoderConfig.radius_out_dim`: `32 → 16`
- `ShellShPoseEncoderConfig.scalar_out_dim`: `32 → 16`
- `ShellShPoseEncoderConfig.scalar_hidden_dim`: `64 → 32`

Notes:
- Kept `frustum_grid_size=4` and `frustum_depths_m=[0.5,1.0,2.0,3.0]` so K remains 64 (docs stay accurate).

### PoseTW-only SE(3) point transforms for frustum samples

File: `oracle_rri/oracle_rri/vin/model.py`

- `_frustum_points_world(...)` now uses `PoseTW * points` for world-point generation (no manual `R/t` math).
- Added `self._frustum_points_cam: Tensor` annotation + a clarifying NOTE about buffer registration.

## Verification

- `cd oracle_rri && uv run ruff format oracle_rri/vin/model.py oracle_rri/vin/spherical_encoding.py`
- `cd oracle_rri && uv run ruff check oracle_rri/vin/model.py oracle_rri/vin/spherical_encoding.py`
- `cd oracle_rri && uv run pytest tests/vin`
- `cd oracle_rri && uv run pytest -m integration tests/integration/test_vin_real_data.py tests/integration/test_vin_lightning_real_data.py`


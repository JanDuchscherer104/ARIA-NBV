# VIN simplification (v0.1 internal refactor) — 2025-12-17

## Goal

Reduce the number of tiny standalone `nn.Module` helpers in `oracle_rri.vin` while keeping the v0.1 architecture and public API stable.

## What changed

- Folded the previously separate v0.1 helper modules into `oracle_rri/oracle_rri/vin/model.py`:
  - Scene-field construction, 1×1×1 field projection, global 6³ token pooling, frustum sampling, and candidate cross-attention pooling are now implemented as:
    - small module-level helper functions (`_build_scene_field`, `_sample_voxel_field`, `_build_frustum_points_cam`), and
    - `VinModel` internal methods (`_pool_global`, `_frustum_points_world`, `_pool_candidates`).
- Removed the now-redundant files:
  - `oracle_rri/oracle_rri/vin/scene_field.py`
  - `oracle_rri/oracle_rri/vin/field_compress.py`
  - `oracle_rri/oracle_rri/vin/global_pool.py`
  - `oracle_rri/oracle_rri/vin/frustum_query.py`
  - `oracle_rri/oracle_rri/vin/candidate_pool.py`
- Flattened `VinModelConfig`:
  - replaced nested config objects for the v0.1 components with a small set of direct fields (scene-field channels, field dim, attention heads, frustum depths, etc.).

## Coordinate-system + validity guardrails

- WORLD→VOXEL mapping uses `voxel/T_world_voxel` (world←voxel) from EVL and is explicitly inverted to voxel←world during sampling.
  - See NOTE/FIXME in `oracle_rri/oracle_rri/vin/model.py` `_sample_voxel_field(...)`.
- Candidate validity is defined as `token_valid.any(dim=-1)` (any frustum sample in-grid).
  - TODO in `oracle_rri/oracle_rri/vin/model.py` suggests optionally AND-ing with “camera center in voxel grid” if you want stricter semantics.

## Tests executed (real data + unit)

- Unit: `cd oracle_rri && uv run pytest tests/vin`
- Integration (real assets): `cd oracle_rri && uv run pytest tests/integration/test_vin_real_data.py -m integration`
- Lightning smoke (real assets): `cd oracle_rri && uv run pytest tests/integration/test_vin_lightning_real_data.py -m integration`

## Follow-up fixes (after refactor)

- Restored `VinLightningModule._fit_binner_from_datamodule()` and added `_resolve_binner_path()` in
  `oracle_rri/oracle_rri/lightning/lit_module.py` after a refactor accidentally commented out the binner fit path.
  - Verified with `tests/integration/test_vin_lightning_real_data.py` (integration).

## Follow-ups / suggestions

- Consider switching `torch.load(..., weights_only=True)` where possible in `rri_binning.py` to silence the FutureWarning (ensure compatibility with older torch).
- If candidate validity semantics become important downstream, add an explicit `center_in_grid` check and log both masks for debugging.

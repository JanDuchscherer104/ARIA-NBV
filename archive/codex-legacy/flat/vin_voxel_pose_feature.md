# VIN voxel pose SH feature

## Change summary

- Added optional voxel-pose encoding to VIN:
  - `voxel/T_world_voxel` is transformed into the reference rig frame (`rig_ref <- voxel`).
  - Encoded with the existing `ShellShPoseEncoder` (SH + radius + scalar alignment).
  - Broadcast as a global token and concatenated to per-candidate features.

## Code touchpoints

- `oracle_rri/oracle_rri/vin/model.py`
  - New config flag: `use_voxel_pose_encoding` (default `True`).
  - Head input dimension now includes an extra `E_pose` when enabled.
  - New debug tensors for voxel pose + encoding.
  - Fixed `scene_field_channels` annotation back to `list[str]` (previous `Literal[...]` broke config parsing).
- `oracle_rri/oracle_rri/vin/types.py`
  - Added `voxel_*` debug fields in `VinForwardDiagnostics`.
- `oracle_rri/oracle_rri/lightning/lit_module.py`
  - `summarize_vin` now prints voxel-pose descriptor shapes.
- `docs/contents/impl/vin_nbv.qmd`
  - Documented voxel-pose encoding and updated concat description.

## Open questions / suggestions

- Consider logging the voxel pose encoding magnitude to ensure it is on a comparable scale to candidate pose encodings.
- If `use_voxel_pose_encoding` becomes optional for ablations, add a small metric to compare runs with and without it (e.g., top‑k recall).

## Test notes

- `tests/vin/test_candidate_validity.py` passed.
- `tests/vin/test_vin_model_integration.py` failed on CPU due to xFormers `memory_efficient_attention` requiring CUDA/FP16. Run on a CUDA machine or disable xFormers in the EVL stack to validate end-to-end.

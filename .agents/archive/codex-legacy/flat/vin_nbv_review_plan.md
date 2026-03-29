# VIN-NBV Review Follow-up (plan + checks)

Date: 2025-12-19

## Findings (from code + docs)
- `VinModel._pool_global` is now a simple voxel-field mean, so the “manual kernel/stride” brittleness called out in the review no longer applies.
- `VinModel.forward` already accepts `reference_pose_world_rig` and uses it when provided (e.g., in `VinLightningModule._step`), so the “reference pose guessing” issue is already addressed for training.
- `lit_datamodule._vin_oracle_batch_from_label` does **not** reindex RRI via `candidate_indices` anymore; it passes `label_batch.rri` as-is. Candidate indices are currently used only for UI/debug tooling.
- The review concern about `CandidateDepths.camera.T_camera_rig` being a *physical* camera extrinsic does not apply here: candidate generation stores **candidate_camera←reference_pose** in `views.T_camera_rig` (see `oracle_rri/oracle_rri/pose_generation/types.py:101`).
- `EvlBackbone` no longer surfaces `free_input` for the minimal VIN scorer to avoid depending on free-space semantics that can change across EVL configs/checkpoints.
- Remaining open risks: decide whether VIN should (a) require `CandidateDepths.p3d_cameras` for frustum sampling, or (b) build its own `PerspectiveCameras` at inference time (no depth rendering) to stay intrinsics-aware.

## Suggested checks (to verify review issues)
- Pose alignment: assert `T_world_cam == T_world_rig @ T_cam_rig.inverse()` for each batch; log max residual.
- Candidate ordering: assert all candidate-wise tensors share the same leading dimension (`poses`, `depths`, `candidate_pcs.lengths`, `rri`).
- Candidate validity: log histogram of `token_valid.float().mean(-1)` to detect overly sparse candidates.
- Binner edges: after fitting, assert `edges` strictly increasing and finite; if not, fall back to a safe default.

## Plan (implemented or in-progress)
1. Add candidate-valid fraction threshold (configurable) and diagnostics.
2. Harden `RriOrdinalBinner._finalize` with finite filtering + unique edges + fallback.
3. Provide an MWE config preset (drop derived/free channels; SH-only pose encoder; optional disable global attention).
4. Add intrinsics-aware frustum sampling using PyTorch3D `PerspectiveCameras.unproject_points` (align with the depth renderer camera model).
5. Add lightweight debug assertions for pose alignment and tensor shape agreement.

## Open questions
- Do you want the MWE to keep a `free_input` channel as an explicit EVL contract (and validate it per checkpoint), or keep it removed entirely until we have end-to-end evidence that it helps?
- Should we default `use_global_pool=False` for the MWE, or keep global attention and only simplify candidate pooling?

## Executed sanity checks (results)
- Binner fit data (`.logs/vin/rri_binner_fit_data.pt`): 3,493 finite samples, edges strictly increasing (0 non-increasing diffs).
  - Saved artifacts: `.logs/vin/rri_binner_edges.npy`, `.logs/vin/rri_binner_rri.npy`, `.logs/vin/rri_binner_fit_data_hist.png`.
- Pose alignment check (real snippet, simplified mesh + downsampled points):
  - `max |T_world_cam - (T_world_rig @ T_cam_rig^{-1})| = 0.0` (passed).
- Candidate ordering check: depth rendering + candidate pointclouds align by leading dimension (tested with 1 candidate; also verified fast path with 2 candidates for depth render).
- Candidate valid fraction (EVL frustum sampling on 4 candidates): mean 0.703, min 0.333, max 0.875 (all non-trivial).
- Free-space channel: when reusing the same model output, `free_input == voxel/feat[:, -1]` (max diff 0.0). Earlier mismatch was due to comparing across two separate EVL forward passes.

## Notes / follow-ups
- RRI computation is slow on CPU unless mesh is aggressively simplified and points are downsampled; GPU would make full-resolution checks feasible.
- Candidate generation seeding (`seed!=None`) may attempt CUDA RNG init even on CPU-only setups; for tests, set `seed=None` to avoid CUDA init errors.

## Fixes applied (code + docs)
- `VinModelConfig`: added `candidate_min_valid_frac` (default 0.2) and use fraction-based validity in `VinModel.forward`.
- `RriOrdinalBinner._finalize`: now filters non-finite samples, de-duplicates quantile edges, and falls back to uniform edges if degenerate.
- `oracle_rri/oracle_rri/vin/model.py`: frustum rays now come from `PerspectiveCameras.unproject_points` when `p3d_cameras` are provided; fallback stays as a fixed-FOV grid.
- `oracle_rri/oracle_rri/vin/backbone_evl.py`: removed `free_input` extraction from `voxel/feat` for the minimal VIN contract.
- `docs/contents/impl/vin_nbv.qmd`: updated EVL/VIN contract notes + frustum sampling description.
- Rendered `docs/contents/impl/vin_nbv.qmd` via Quarto to refresh the HTML output.

## Tests
- `pytest tests/vin/test_rri_binning.py tests/vin/test_candidate_validity.py`

## VIN simplification (MWE) applied
- Removed LFF pose encoder and pose encoding mode toggle; VIN is SH-only.
- Removed global/candidate attention blocks; candidate pooling is now masked mean.
- Scene field defaults to minimal channels: `occ_pr`, `occ_input`, `counts_norm`.
- Global pooling is mean over voxel field (no attention).
- Dropped voxel↔rig link descriptor from concatenated features.
- Updated `oracle_rri/scripts/summarize_vin.py` to match simplified inputs.
- Updated `docs/contents/impl/vin_nbv.qmd` to describe simplified architecture.

## Tests
- `pytest tests/vin/test_vin_model_integration.py`
- Updated `docs/contents/todos.qmd` VIN scorer description to match simplified architecture.

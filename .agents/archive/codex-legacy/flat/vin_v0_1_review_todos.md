# VIN v0.1 (review-driven) TODOs — 2025-12-17

## Context

- Goal: move VIN from **neck-feature center sampling + global mean/max** to a **head-derived low-dim scene field**, **coarse 6³ global tokens**, and a **pose-conditioned frustum query (K≈64) with cross-attention pooling**.
- Repo state (verified locally via `make context` on 2025-12-17):
  - **LFF init bug**: `LearnableFourierFeatures` uses `gamma**2` scaling in `oracle_rri/oracle_rri/vin/pose_encoding.py`.
  - **Backbone contract**: `oracle_rri/oracle_rri/vin/types.py::EvlBackboneOutput` currently only exposes neck feats + voxel pose/extent.
  - **Datamodule**: `oracle_rri/oracle_rri/lightning/lit_datamodule.py` does not expose `CandidateDepths.candidate_indices` in `VinOracleBatch`.
  - **Index nuance**: `CandidateDepths.candidate_indices` are indices into the **full shell (pre-pruning)**, not indices into `CandidateSamplingResult.views` (valid-only).

## P0 fixes (do first)

- [ ] Fix LFF gamma init bug: change `Wr` init scale from `gamma**2` → `gamma` in `oracle_rri/oracle_rri/vin/pose_encoding.py` (`LearnableFourierFeatures`).
- [ ] Add a unit test for the LFF init scale (seeded; large parameter shape; assert `Wr.std()` ≈ `gamma`).
- [ ] Improve `VinOracleBatch` debug alignment:
  - [ ] Add `candidate_indices_global: Tensor["N", int64]` from `label_batch.depths.candidate_indices`.
  - [ ] (Optional) add `candidate_indices_local: Tensor["N", int64]` into `candidates.views` by mapping global→local via `candidates.mask_valid` + `torch.searchsorted` (debug-only).
  - [ ] Add `candidate_depth_valid` or `candidate_valid_pixel_count` derived from `depths_valid_mask` to mask zero-hit candidates (possible when `num_total <= max_final` in `CandidateDepthRenderer`).
- [ ] Update `VinLightningModule` masking to include depth-valid candidates in addition to `candidate_valid` + finite RRI.

## Backbone contract (heads vs neck)

- [ ] Expand `oracle_rri/oracle_rri/vin/types.py::EvlBackboneOutput` with optional head/evidence tensors:
  - `occ_pr: Tensor | None` (B,1,D,H,W)
  - `occ_input: Tensor | None` (B,1,D,H,W)
  - `free_input: Tensor | None` (B,1,D,H,W) (from `out["voxel/feat"][:, -1:]`)
  - `counts: Tensor | None` (B,D,H,W) int64
  - `counts_m: Tensor | None` (B,D,H,W) int64 (optional/debug)
  - keep `occ_feat/obb_feat` for ablations.
- [ ] Add `EvlBackboneConfig.features_mode: Literal["heads","neck","both"] = "heads"` in `oracle_rri/oracle_rri/vin/backbone_evl.py`.
- [ ] Modify `EvlBackbone.forward` to extract and return the head/evidence tensors:
  - `out["occ_pr"]`, `out["voxel/occ_input"]`, `out["voxel/counts"]`, `out["voxel/counts_m"]`, `out["voxel/feat"][:, -1:]`.
  - Populate fields according to `features_mode`, validate shapes/dtypes, and keep `t_world_voxel` + `voxel_extent` as the coordinate contract.

## v0.1 VIN architecture modules

- [ ] Add `oracle_rri/oracle_rri/vin/scene_field.py`:
  - `SceneFieldBuilder(nn.Module)` + config.
  - Build `field: Tensor[B,C,D,H,W]` from `occ_pr/occ_input/free_input/counts` and derived masks (e.g., `unknown`, `counts_norm`, `new_surface_prior`).
  - Confirm whether `occ_pr` is already a probability (vendored EVL sets `occ_pr = sigmoid(logits)`); avoid double-sigmoid; keep a safety switch if needed.
- [ ] Add `oracle_rri/oracle_rri/vin/field_compress.py`:
  - `FieldCompressor(nn.Module)` = `Conv3d(1×1×1) -> GroupNorm -> GELU` with config (`d0`, `gn_groups`).
- [ ] Add `oracle_rri/oracle_rri/vin/global_pool.py`:
  - `GlobalTokenPooler(nn.Module)` with:
    - `adaptive_avg_pool3d(field, (6,6,6)) -> tokens (B,216,d0)`,
    - learnable query vectors + `nn.MultiheadAttention` pooling,
    - optional lightweight 3D positional encoding.
- [ ] Add `oracle_rri/oracle_rri/vin/frustum_query.py`:
  - `FrustumPointSampler` (module or pure function) generating camera-frame points (K=dirs×depths; default 16×4=64) from a simple grid + depth list; configurable `fov_deg`.
- [ ] Add `oracle_rri/oracle_rri/vin/candidate_pool.py`:
  - `VoxelSampler`: world→voxel coords via `T_world_voxel^{-1}` + `voxel_extent`, flatten `(B,N,K,3)`→`(B,N*K,3)` and sample with `efm3d.utils.voxel_sampling.sample_voxels`, reshape back.
  - `CandidateTokenPooler(nn.Module)`: pose→query projection and cross-attention pooling over K tokens with `key_padding_mask` from the validity mask.

## `VinModel` refactor

- [ ] Update `oracle_rri/oracle_rri/vin/model.py` to use:
  - `backbone (heads mode) -> SceneFieldBuilder -> FieldCompressor`,
  - `GlobalTokenPooler` for global context,
  - `ShellShPoseEncoder` for pose embedding (keep `lff6d` baseline but ensure the init bugfix),
  - `FrustumPointSampler -> VoxelSampler -> CandidateTokenPooler` for candidate embedding,
  - final scorer head: concat `[pose_embed, global_embed, cand_embed] -> MLP -> CORAL`.
- [ ] Update `VinModel.forward` to accept either:
  - training path: `candidate_poses_camera_rig` + `reference_pose_world_rig` (compute `candidate_poses_world_cam` internally), or
  - inference path: `candidate_poses_world_cam` (derive reference from the snippet).
- [ ] Update `VinPrediction` validity semantics:
  - `candidate_valid = token_valid.any(dim=-1)` (optionally AND center-in-grid),
  - optionally return `token_valid` for debugging.

## Training + tests

- [ ] Update `oracle_rri/oracle_rri/lightning/lit_datamodule.py` for new `VinOracleBatch` fields and masking (do **not** index `candidates.views` with `depths.candidate_indices` without global→local mapping; indices are global).
- [ ] Update integration tests:
  - `oracle_rri/tests/integration/test_vin_real_data.py` (shape assertions + new `candidate_valid` semantics),
  - `oracle_rri/tests/integration/test_vin_lightning_real_data.py` (smoke `trainer.fit` still runs).
- [ ] Add/extend integration smoke: one real snippet → oracle labeler → VIN forward → CORAL loss finite.
- [ ] Ensure `ruff format`, `ruff check`, and targeted `pytest` pass for all touched files.

## Docs

- [ ] Update `docs/contents/impl/vin_nbv.qmd`:
  - Clarify EVL out-dict vs `EvlBackboneOutput` contract,
  - document the new scene-field channels, global token pooling, frustum query, and attention guardrail.

## Acceptance criteria (ship v0.1)

- [ ] `pytest oracle_rri/tests/integration/test_vin_real_data.py -m integration` passes on a real local scene.
- [ ] `pytest oracle_rri/tests/integration/test_vin_lightning_real_data.py -m integration` passes (`fast_dev_run`).
- [ ] Minimal training smoke `cd oracle_rri && uv run python scripts/train_vin.py --max-steps 5 ...` runs without NaNs and logs finite loss.
- [ ] Guardrail met: attention only over (a) coarse 6³ tokens and (b) local K frustum samples (no attention over full 48³).


---
id: 2026-01-03_review_model_v2_offline_only_2026-01-03
date: 2026-01-03
title: "Review Model V2 Offline Only 2026 01 03"
status: legacy-imported
topics: [model, v2, offline, only, 2026]
source_legacy_path: ".codex/review_model_v2_offline_only_2026-01-03.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Review: `model_v2.py` + `offline_only.toml` (2026-01-03)

## Scope

- `oracle_rri/oracle_rri/vin/model_v2.py`
- `.configs/offline_only.toml`

## Quick validation

- `tomllib` parses `.configs/offline_only.toml` and it loads into `AriaNBVExperimentConfig` cleanly.
- `ruff check oracle_rri/oracle_rri/vin/model_v2.py` passes.

## `model_v2.py` review

### Strengths

- Clear architecture docstring and modular private helpers (`_prepare_inputs`, `_encode_pose_features`, `_build_field_bundle`, …).
- Good frame hygiene: pose features and positional keys are computed in the **reference rig frame**, with an explicit CW90 undo option.
- Semidense projection features (`coverage`, `empty_frac`, `valid_frac`, `depth_mean`, `depth_std`) are *roll-invariant* by construction (grid occupancy fraction is invariant to 90° bin permutation), which is a nice property when `rotate_yaw_cw90` is present in the pipeline.

### Issues / risks

1) **CW90 correction doc mismatch (poses vs cameras).**
   - `VinModelV2Config.apply_cw90_correction` says it undoes CW90 on “poses + cameras”, but the implementation only undoes it for `pose_world_cam` and `pose_world_rig_ref`.
   - This is likely OK with `enable_semidense_frustum=false` (default), because the projection summary features are roll-invariant. If `enable_semidense_frustum=true`, `x_norm/y_norm` tokens become roll-sensitive and you probably want camera/pose conventions aligned.

2) **Trajectory attention normalization channel dim can be wrong when `traj_dim != pose_dim`.**
   - `traj_attn` outputs `embed_dim == pose_dim`, but `traj_attn_norm` is instantiated with `num_channels=traj_dim`.
   - With the current `offline_only.toml`, pose/traj encoders both output 32 so this doesn’t break, but it’s fragile for future configs.

3) **Minor duplication in `summarize_vin`.**
   - `feature_summary["semidense_proj"]` is set twice (harmless but noisy).

4) **Hard assumptions about backbone outputs.**
   - `_build_field_bundle` assumes `occ_pr`, `cent_pr`, `occ_input`, `counts` are present; if a backbone config changes `features_mode`, failures will be attribute errors rather than a clear message.

## `offline_only.toml` review

### Consistency with `VinModelV2`

- `datamodule_config.efm_keep_keys` includes the semidense keys needed by `EfmPointsView.collapse_points(... include_inv_dist_std=True)` and the trajectory keys needed by `VinModelV2._encode_traj_features`.
- `module_config.vin.scene_field_channels` matches the channels constructed in `VinModelV2._build_field_bundle`.

### Operational notes

- `trainer_config.use_wandb=true` will require W&B connectivity unless you explicitly run with `WANDB_MODE=offline` / `WANDB_DISABLED=true`.
- Consider `persistent_workers=true` when training long runs from cache to avoid worker restart overhead (only if memory is stable).

## Suggested follow-ups (optional patches)

- Fix `traj_attn_norm` to normalize `pose_dim` channels (or switch to `LayerNorm` on the last dim).
- Either update the `apply_cw90_correction` docstring to “poses only”, or implement camera correction when `enable_semidense_frustum` is enabled.
- Remove duplicate `semidense_proj` entry in `summarize_vin`.

---
id: 2026-01-05_mock_optuna_sweep_2026-01-05
date: 2026-01-05
title: "Mock Optuna Sweep 2026 01 05"
status: legacy-imported
topics: [mock, optuna, sweep, 2026, 01]
source_legacy_path: ".codex/mock_optuna_sweep_2026-01-05.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Mock Optuna Sweep (VIN cache) — 2026-01-05

## Config + run
- Config: `.configs/offline_cache_required_one_step_vin_cache.toml`
  - `run_mode="optuna"`, `batch_size=2`, `limit_train_batches=10`, `limit_val_batches=0`
  - `datamodule_config.source.cache.limit=20`
  - `vin_snippet_cache_allow_subset=true` (required for partial VIN cache)
  - `optuna_config.study_name="vin-v2-mock-sweep"`, `n_trials=10`, `monitor="train/loss"`
- Command: `oracle_rri/.venv/bin/nbv-optuna --config-path ./.configs/offline_cache_required_one_step_vin_cache.toml`

## Outcome
- Study reused existing storage (`load_if_exists=true`), total trials now: 13
- Best value: `train/loss=7.357858657836914`
- Best params:
  - `module_config.vin.use_point_encoder=True`
  - `module_config.vin.use_traj_encoder=True`
  - `module_config.vin.semidense_obs_count_norm="none"`
  - `module_config.vin.semidense_visibility_embed=True`
  - `module_config.vin.semidense_frustum_mask_invalid=True`
  - `module_config.vin.use_voxel_valid_frac_feature=True`
  - `module_config.vin.use_voxel_valid_frac_gate=True`
  - `module_config.vin.global_pool_grid_size=8`
  - `module_config.optimizer.learning_rate=1.646292492406435e-05`
  - `module_config.optimizer.weight_decay=0.0003515381015917694`

## Notes
- Trial count exceeded the 10 “mock” trials because the study resumed from previous attempts.
- Keep `vin_snippet_cache_allow_subset=true` while the VIN cache is incomplete, else missing-snippet errors return.

## Update (Sweep config + W&B tags)
- Updated `.configs/sweep_config.toml` to align with `offline_only.toml` defaults (OneCycleLR schedule, aux regression decay, coverage weighting) and set `batch_size=16` with `max_epochs=3` (curriculum/coverage anneal over 3 epochs).
- `oracle_rri/oracle_rri/lightning/aria_nbv_experiment.py` now injects W&B tag `optuna` for optuna runs (alongside group/job_type).
- Adjusted `gradient_clip_val` to `5.0` in `.configs/sweep_config.toml` based on observed gradnorm spikes.
- Added `enable_semidense_frustum` as an Optuna sweep parameter via `optimizable_field` in `oracle_rri/oracle_rri/vin/model_v2.py`.
- Backfilled Optuna objective values in `.logs/optuna/vin-v2-sweep.db` from W&B `val/coral_loss_rel_random` (21 trials updated, INF cleared).
- Improved Optuna metric retrieval in `oracle_rri/oracle_rri/lightning/aria_nbv_experiment.py` to try `monitor`, `monitor_epoch`, `monitor_step`, and base key fallbacks so metrics logged with Lightning suffixes are found.
- Updated `.configs/sweep_config.toml` to include missing child configs from `offline_only.toml` (trainer callbacks/options, accelerator/devices, fast_dev_run) and added the `module_config.vin.point_encoder` block so `use_point_encoder` actually toggles the encoder. Batch size set to 16; log_interval_steps set to null.
- T22 W&B config shows `use_point_encoder=True` but `point_encoder=None`; model summary lacks PointNeXt encoder as expected. The sweep config must include the `module_config.vin.point_encoder` block for this toggle to be effective (now fixed).
- Corrected `.logs/optuna/vin-v2-sweep.db`: set all `module_config.vin.use_point_encoder` trial params to `false` (0.0) because the sweep lacked `point_encoder` config; added `config_correction` user attribute to all trials noting the fix.
- Added `relies_on` support to `Optimizable` + Optuna config traversal; semidense frustum-dependent params now only optimize when `enable_semidense_frustum=True`. Corrected optuna DB to force `semidense_visibility_embed`/`semidense_frustum_mask_invalid` false when frustum disabled and tagged trials with `config_correction_frustum_deps`.
- Corrected Optuna DB again: `module_config.vin.use_point_encoder` set to index=1 (false) for all trials; added `config_correction_point_encoder` attribute noting categorical choices=[true,false].

## Update (Semidense sweep knobs + OneCycle LR)
- Re-enabled Optuna knobs in `oracle_rri/oracle_rri/vin/model_v2.py` for `head_hidden_dim`, `head_num_layers`, `head_dropout`, `field_dim`, `field_gn_groups`.
- Added `relies_on` gating for semidense/point-encoder-related params (now only optimized when `module_config.vin.use_point_encoder=True`):
  - `semidense_proj_grid_size`, `semidense_proj_max_points`, `semidense_frustum_max_points`
  - `enable_semidense_frustum`, `semidense_include_obs_count`, `semidense_obs_count_norm`
  - `semidense_visibility_embed`, `semidense_frustum_mask_invalid` (also require `enable_semidense_frustum=True`).
- Made PointNeXt encoder knobs optimizable in `oracle_rri/oracle_rri/vin/pointnext_encoder.py`: `out_dim`, `max_points` (both gated on `use_point_encoder=True`).
- Added OneCycleLR sweep knobs in `oracle_rri/oracle_rri/lightning/optimizers.py`: `max_lr`, `div_factor`, `final_div_factor`, `pct_start` (and gated AdamW `learning_rate` on `lr_scheduler.max_lr=None`).

## Test status
- `oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_v2_utils.py` failed:
  - `SEMIDENSE_PROJ_FEATURES` lacks `"valid_frac"`.
  - `proj_data` missing `"finite"` key in `_encode_semidense_projection_features`.
- `uv run pytest tests/vin/test_vin_v2_utils.py` failed due to missing `efm3d` (system Python 3.12). Use the venv instead.

## Update (Semidense candidate vis rename)
- Renamed semidense projection feature to `semidense_candidate_vis_frac` and added backward-compatible aliases for `valid_frac`/`semidense_valid_frac`.
- `VinPrediction`/`VinV2ForwardDiagnostics` now carry `semidense_candidate_vis_frac` plus deprecated `semidense_valid_frac` alias.
- Logging now emits both new and legacy metric keys.
- Tests updated to index the renamed projection feature.
- Tests: `oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_v2_utils.py` ✅

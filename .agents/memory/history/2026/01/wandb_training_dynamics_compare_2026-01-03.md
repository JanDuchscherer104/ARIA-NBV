---
id: 2026-01-03_wandb_training_dynamics_compare_2026-01-03
date: 2026-01-03
title: "Wandb Training Dynamics Compare 2026 01 03"
status: legacy-imported
topics: [wandb, training, dynamics, compare, 2026]
source_legacy_path: ".codex/wandb_training_dynamics_compare_2026-01-03.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# W&B run comparison: training dynamics (2026-01-03)

Metric: `train/coral_loss_rel_random_step` (lower is better; normalized by random baseline `(K-1)*log(2)`)

## Update

- Fixed `.configs/offline_only.toml` `module_config.lr_scheduler.final_div_factor` from `1e-4` → `1e4`.
  - `OneCycleLR` uses `final_lr = max_lr / (div_factor * final_div_factor)`; values `< 1` make the LR ramp *up* at the end of the cycle (can destabilize training).

## LR analysis from `.logs/wandb_export_2026-01-03T16_14_40.211+01_00.csv`

At `trainer/global_step = 279` the logged `lr-AdamW` values are:

- OneCycle runs (warmup): `~5e-05`
  - `JanPC`: `4.656e-05`
  - `fix-masks` / `coral-fixed-masks` / `bce+frustum-sampling` / `one-cycle-higher-grad-clip`: `5.088e-05`
- Plateau / fixed-LR runs: `5e-04` or `1e-03`
  - `fixesIII`, `latest+fixes-2`: `5e-04`
  - `traj-ecnoder+8classes`, `no-valid+aux-loss-decay+traj-enc`: `1e-03`

Within the OneCycle group, `JanPC` warms up **more slowly** than `fix-masks`:

- Step `1500`: `JanPC = 2.179e-04` vs `fix-masks = 3.225e-04` (≈ `+48%`).
- Step `2000`: `JanPC = 3.402e-04` (still below the `5e-04` plateau runs).
- Step `2500`: `JanPC = 4.782e-04` (now comparable to `5e-04` plateau runs).
- Step `2800`: `JanPC = 5.635e-04`.

Interpretation:

- Yes — with `OneCycleLR`, the *early* training dynamics can look “slower” simply because the LR is much smaller than in fixed/plateau runs.
- With `max_epochs = 50` and `pct_start = 0.15`, the warmup spans ~`7.5` epochs, so after ~`4` epochs you’re still in the warmup phase.

## Suggested `offline_only.toml` tweaks (if you want faster early dynamics)

- Shorten warmup so `max_lr` is reached by ~epoch 4:
  - Set `module_config.lr_scheduler.pct_start = 0.08` (since `4/50 = 0.08`).
- Increase the initial LR (reduce “slow start” effect):
  - Set `module_config.lr_scheduler.div_factor = 10` (initial LR becomes `max_lr/10` instead of `max_lr/25`).
- Reduce loss spikes while LR ramps up:
  - Set `trainer_config.gradient_clip_val = 3.0` (or `4.0`) instead of `8.0`.
- Keep `module_config.lr_scheduler.final_div_factor = 1e4` (prevents end-of-cycle LR blow-up).

## Metric snapshots
|run|state|@500|@1000|@1400|@2000|min<=@2000|
|---|---|---|---|---|---|---|
|JanPC (7lcezvkd)|running|0.895|1.285|0.684|0.718|0.447@1952|
|fix-masks (jejo31ut)|failed|0.740|0.794|0.828|1.040|0.472@690|
|latest+fixes-2 (60s8bkp7)|failed|0.839|0.885|0.764|0.659|0.603@1895|
|traj-ecnoder+8classes (z9kjxmor)|failed|0.955|0.653|0.877|0.616|0.510@1222|
|no-valid+aux-loss-decay+traj-enc (03g7ef8k)|failed|0.995|0.690|0.856|0.587|0.494@1222|
|coral-fixed-masks (m06auwmr)|finished|1.210|1.210|1.210|1.210|0.472@267|

## Key hyperparameter deltas (vs local `.configs/offline_only.toml`)
### JanPC (7lcezvkd)
- (no differences found for selected keys; run likely matches current config or keys missing)

### fix-masks (jejo31ut)
- (no differences found for selected keys; run likely matches current config or keys missing)

### latest+fixes-2 (60s8bkp7)
- `module_config.optimizer.learning_rate`: current=0.0003 vs run=0.0005
- `module_config.aux_regression_weight_gamma`: current=0.9 vs run=0.99
- `module_config.num_classes`: current=15 vs run=30
- `module_config.vin.scene_field_channels`: current=occ_pr, occ_input, counts_norm, cent_pr, free_input, new_surface_prior vs run=occ_pr, occ_input, counts_norm, cent_pr

### traj-ecnoder+8classes (z9kjxmor)
- `module_config.optimizer.learning_rate`: current=0.0003 vs run=0.001
- `module_config.aux_regression_weight_gamma`: current=0.9 vs run=0.99
- `module_config.num_classes`: current=15 vs run=8
- `module_config.vin.field_dim`: current=16 vs run=32
- `module_config.vin.head_hidden_dim`: current=192 vs run=128
- `module_config.vin.head_num_layers`: current=3 vs run=1
- `module_config.vin.head_dropout`: current=0.05 vs run=0
- `module_config.vin.scene_field_channels`: current=occ_pr, occ_input, counts_norm, cent_pr, free_input, new_surface_prior vs run=occ_pr, cent_pr, counts_norm

### no-valid+aux-loss-decay+traj-enc (03g7ef8k)
- `module_config.optimizer.learning_rate`: current=0.0003 vs run=0.001
- `module_config.aux_regression_weight`: current=10 vs run=1
- `module_config.aux_regression_weight_gamma`: current=0.9 vs run=0.98
- `module_config.vin.field_dim`: current=16 vs run=32
- `module_config.vin.head_hidden_dim`: current=192 vs run=128
- `module_config.vin.head_num_layers`: current=3 vs run=1
- `module_config.vin.head_dropout`: current=0.05 vs run=0
- `module_config.vin.scene_field_channels`: current=occ_pr, occ_input, counts_norm, cent_pr, free_input, new_surface_prior vs run=occ_pr, cent_pr, counts_norm

### coral-fixed-masks (m06auwmr)
- `module_config.lr_scheduler.final_div_factor`: current=0.0001 vs run=10000

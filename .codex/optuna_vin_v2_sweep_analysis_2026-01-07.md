# Optuna sweep analysis — `vin-v2-sweep` (`.logs/optuna/vin-v2-sweep.db`)

## Objective / scope

Analyze the Optuna study results in `.logs/optuna/vin-v2-sweep.db` and summarize:

- best trials + their hyperparameters,
- what the sweep *actually* explored (and what it did not),
- inconsistencies that make Optuna importances / comparisons unreliable,
- concrete recommendations for the next sweep round.

## Snapshot

- Study: `vin-v2-sweep` (direction: **minimize**)
- Trials in DB: **35**
  - **30 COMPLETE** (finite objective values)
  - **5 FAIL** (`0, 2, 3, 34, 35`)
- Objective distribution over COMPLETE trials:
  - min **0.724706** (trial **20**)
  - median **0.729601**
  - mean **0.735660**
  - max **0.763974**

Exports:

- `.codex/optuna_vin_v2_trials_raw.csv`: full trial dump (including FAILED)
- `.codex/optuna_vin_v2_best_trial.json`: best trial + params

## Best trials (COMPLETE)

Top 5 by objective:

| trial | objective | use_traj | frustum | obs_count | vis_embed | mask_invalid | voxel_gate | voxel_feat | pool_grid | lr | wd |
|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|
| 20 | 0.724706 | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | 8 | 3.50e-05 | 3.25e-03 |
| 23 | 0.724820 | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | 4 | 1.82e-04 | 7.87e-02 |
| 8  | 0.724854 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 5 | 1.16e-04 | 5.67e-03 |
| 9  | 0.725020 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 5 | 4.50e-05 | 1.93e-02 |
| 11 | 0.725633 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 5 | 2.30e-05 | 3.92e-04 |

Notes:

- `use_point_encoder` is **False** for all top trials above.
- `enable_semidense_frustum=True` in all top trials above.

## Critical caveat: the study is **non-stationary** (config changed mid-study)

Optuna comparisons assume trials are drawn from the *same* objective function / baseline configuration.
That is **not true** here.

### Evidence: “config_correction” flags in trial metadata

There are **23** trials with `trial.user_attrs["config_correction"]` set.
All of these have `module_config.vin.use_point_encoder=False`.

This indicates that, for a large prefix of the study, the runtime code detected a mismatch (missing point-encoder config)
and **forced** `use_point_encoder=False` (and sometimes also forced semidense-frustum-dependent flags).

Practical consequence:

- Most “good” trials (~0.724–0.731) come from the **corrected** regime where the point encoder was not allowed to turn on.
- Later trials (without corrections) include `use_point_encoder=True`, but their objectives live on a *different scale* (~0.733–0.764).

So any global parameter importance / aggregated boolean “effects” across *all* COMPLETE trials is confounded.

### Two regimes in the same study

If you split COMPLETE trials by `has_config_correction`:

- **Corrected regime** (`has_config_correction=True`, n=20):
  - objective mean **0.7278**, std **0.0020**
  - point encoder effectively **constant off**
- **Uncorrected regime** (`has_config_correction=False`, n=10):
  - objective values roughly **0.734–0.764** (much worse)
  - point encoder sometimes **on**

These should not be compared as “same sweep”.

## What Optuna likely learned (within the corrected regime only)

Because the corrected regime is internally consistent (tight objective distribution), it’s the only subset where the
ranking is trustworthy.

Within `has_config_correction=True` (n=20):

- `use_traj_encoder=True` performs better on average than `False` (~0.003 objective gap; balanced counts 10 vs 10).
- `enable_semidense_frustum=True` is *slightly* better than `False` (~0.001 gap; n=15 vs 5).
- `global_pool_grid_size`: smaller grids (4/5/8) cluster among best values; 6/7 look slightly worse (differences are tiny).
- `semidense_include_obs_count`: very small effect in this data (10 vs 10).

I would treat all of these as *weak signals* because the absolute spread is small and the study is short (3 epochs).

## Point encoder: why it “doesn’t make a difference” (from this DB)

This DB cannot answer the point-encoder question cleanly:

1) In the corrected regime (the majority of good trials), `use_point_encoder` was forced **off**, so Optuna did not explore it.
2) In the later regime where `use_point_encoder` toggles, the baseline seems to have changed (objectives are worse overall),
   so comparing “point encoder on/off” there does not transfer to the earlier regime.

Recommendation: treat the current study as invalid evidence for “PointNeXt helps/doesn’t help”.

## Recommendations for the next sweep round

### 1) Make trials comparable (most important)

Pick one baseline (ideally the current best trial’s *config version*) and keep it fixed:

- Same dataset/cache setup
- Same loss / curriculum / coverage weighting config
- Same LR scheduler config

If you want to keep the same Optuna study name, at least tag trials (e.g., `trial.set_user_attr("sweep_phase", "...")`)
so analysis can filter later. Otherwise, start a new study name (cleaner).

### 2) Stop sweeping `optimizer.learning_rate` if OneCycle `max_lr` is fixed

In `.configs/sweep_config.toml`, `module_config.lr_scheduler.max_lr` is set (2e-4).
PyTorch `OneCycleLR` drives the per-step LR from `max_lr/div_factor`, so the optimizer LR is mostly irrelevant.

If you want Optuna to tune LR, either:

- sweep `module_config.lr_scheduler.max_lr` (and maybe `div_factor` / `pct_start`), or
- set `max_lr = null` and define “optimizer.lr is max_lr” as the policy.

### 3) Run a dedicated “point encoder ablation” batch

To answer whether PointNeXt helps:

- freeze all other knobs to the best baseline,
- run (at least) ~5 seeds × {point encoder on/off},
- log objective + calibration metrics (Spearman, confusion hist, monotonicity violations).

If the point encoder hurts, the next knob to add is an explicit *strength/gate* for the point-conditioned FiLM path,
so PointNeXt can’t dominate early training.


---
id: 2026-01-26_vin_v3_vs_best_trials_2026-01-26
date: 2026-01-26
title: "Vin V3 Vs Best Trials 2026 01 26"
status: legacy-imported
topics: [v3, vs, best, trials, 2026]
source_legacy_path: ".codex/vin_v3_vs_best_trials_2026-01-26.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VINv3 vs best trials (2026-01-26)

## Sweep context
- Sweep analysis: `.codex/optuna_vin_v2_sweep_analysis_2026-01-07.md`.
- Best Optuna DB trial: #20 (objective 0.724706) → point encoder effectively OFF (corrected regime).
- User-designated best W&B run: `wsfpssd8` (T41) → point encoder ON, traj encoder ON, semidense frustum ON.
- Study is **non-stationary** due to mid-sweep config corrections; point encoder effect is not trustworthy.

## Architecture mismatches vs best runs
- v3 removes: semidense frustum MHCA, trajectory encoder, point encoder.
- Best trials in corrected regime show weak positive signal for:
  - `use_traj_encoder=True`
  - `enable_semidense_frustum=True`
  - `use_voxel_valid_frac_feature=True` (gate OFF)
- `wsfpssd8` specifically uses point encoder + traj + frustum + voxel-valid feature (gate OFF).
- v3 only FiLM-modulates global features with semidense projection stats (no concat), reducing semidense signal strength compared to v2 best.

## Potential issues in current v3 implementation
- **CW90 correction mismatch**: `apply_cw90_correction` rotates poses only; `p3d_cameras` remain unrotated → semidense projections can be inconsistent.
- **Semidense reliability stats**: v3 weights by global `semidense_obs_count_*` / `semidense_inv_dist_std_*` defaults; if stale vs cache, weights saturate and collapse semidense signal.
- **Gating difference**: v3 defaults `use_voxel_valid_frac_gate=True` and no valid-frac feature; best trials used gate OFF + valid-frac feature concat. Gating can suppress global features when candidates are OOB.
- **Signal path reduction**: removing frustum + traj reduces per-candidate discriminative cues that were slightly favored in sweep.
- **VinSnippetView NaNs**: v3 relies on `lengths` and does not filter non-finite points for VinSnippetView; if cache padding contains NaNs and lengths are wrong, projections degrade.

## File references
- v3 implementation: `oracle_rri/oracle_rri/vin/model_v3.py`.
- v2 implementation: `oracle_rri/oracle_rri/vin/experimental/model_v2.py`.
- Sweep analysis: `.codex/optuna_vin_v2_sweep_analysis_2026-01-07.md`.
- Best DB trial: `.codex/optuna_vin_v2_best_trial.json`.

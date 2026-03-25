---
id: 2026-01-26_vin_v3_vs_v2_mismatch_2026-01-26
date: 2026-01-26
title: "Vin V3 Vs V2 Mismatch 2026 01 26"
status: legacy-imported
topics: [v3, vs, v2, mismatch, 2026]
source_legacy_path: ".codex/vin_v3_vs_v2_mismatch_2026-01-26.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VINv3 vs v2 best-run mismatch review (2026-01-26)

## Scope
- Compared W&B configs for the first VINv3 run and the best optuna VIN v2 run.
- Focused on architectural feature deltas and training hyperparameters.

## Runs compared
- VINv3 run: `jzddfu6u` (name: `vin-v3-01`, created 2026-01-07).
- Best VIN v2 run (optuna): `wsfpssd8` (name: `R2026-01-07_12-16-12_T41`).

## Key architecture mismatches
- **Dropped modules in v3**:
  - v2 uses `use_traj_encoder=True` (trajectory context) → v3 has no traj encoder support.
  - v2 uses `use_point_encoder=True` (PointNeXt semidense encoder) → v3 removes point encoder.
  - v2 uses `enable_semidense_frustum=True` (frustum MHCA) → v3 has no frustum attention.
- **Semidense features**:
  - v2 concatenates projection + frustum features into the head; v3 only FiLM-modulates global features.
  - v2 uses `semidense_visibility_embed=True` and `semidense_include_obs_count=True`; v3 replaces this with global obs_count/inv_dist_std weighting.
- **Voxel valid fraction**:
  - v2 sets `use_voxel_valid_frac_gate=False` and `use_voxel_valid_frac_feature=True` (explicit feature concat).
  - v3 sets `use_voxel_valid_frac_gate=True` and has no valid-frac feature, changing how low-coverage candidates are handled.
- **Head/field capacity**:
  - `field_dim`: v2=24 vs v3=16 (smaller voxel field).
  - `head_num_layers`: v2=1 vs v3=3 (deeper head in v3).
  - `head_dropout`: v2≈0.007 vs v3=0.05.
  - `global_pool_grid_size`: v2=5 vs v3=6.
  - `semidense_proj_grid_size`: v2=12 vs v3=16.

## Training hyperparameter mismatches
- `batch_size`: v2=8 vs v3=4.
- `num_workers`: v2=15 vs v3=12.
- Optimizer weight decay: v2≈0.0132 vs v3=0.001.
- OneCycleLR settings differ (max_lr, div_factor, final_div_factor, pct_start).

## Implementation correctness risk (likely impacts semidense signal)
- `apply_cw90_correction=True` in both runs, but `VinModelV3` only rotates poses; it does **not** rotate `p3d_cameras` used for projection. This can misalign pose/camera frames and corrupt semidense features.

## Suggested follow-ups
- Re-run v3 with v2-best architectural signals restored (traj encoder, point encoder, semidense frustum) or add a “v3-compat” mode that concatenates semidense stats into the head.
- Align v3 hyperparameters with v2-best (field_dim=24, head_num_layers=1, head_dropout~0.007, global_pool_grid_size=5, semidense_proj_grid_size=12) for an apples-to-apples comparison.
- Fix or assert CW90 camera consistency before relying on semidense projection features.
- Revisit voxel-valid gating (disable gate, re-add valid-frac feature) to reduce collapse risk.

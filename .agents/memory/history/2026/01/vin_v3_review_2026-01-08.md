---
id: 2026-01-08_vin_v3_review_2026-01-08
date: 2026-01-08
title: "Vin V3 Review 2026 01 08"
status: legacy-imported
topics: [v3, 2026, 01, 08]
source_legacy_path: ".codex/vin_v3_review_2026-01-08.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VIN v3 Review (2026-01-08)

## Scope
- Reviewed `oracle_rri/oracle_rri/vin/model_v3.py` and compared against `oracle_rri/oracle_rri/vin/experimental/model_v2.py` using the provided v2 config snapshot.
- Focused on potential correctness risks, mismatches, and functional deltas that could affect training or diagnostics.

## Key findings (potential issues)
1) **CW90 correction mismatch with cameras**
   - `VinModelV3` undoes `rotate_yaw_cw90` for poses but does not adjust `p3d_cameras`, yet relies on `p3d_cameras` for semidense and voxel projections. This can misalign candidate pose frames vs. camera frames and corrupt semidense features.
   - This same mismatch exists in v2 (`apply_cw90_correction` claims to correct cameras but only touches poses).
   - References: `oracle_rri/oracle_rri/vin/model_v3.py` docstring and `_prepare_inputs` (lines ~38–46, ~359–362); `oracle_rri/oracle_rri/vin/experimental/model_v2.py` config docstring and `_prepare_inputs` (lines ~373–374, ~622–624).

2) **VinSnippetView semidense sampling does not filter invalid/NaN points in v3**
   - v3’s `_sample_semidense_points` uses `lengths` or raw slices but does not filter non-finite points. If cache padding uses NaNs (or lengths is absent/stale), NaNs propagate into projection, reducing valid counts and skewing coverage/visibility stats.
   - v2 explicitly filters finite points in the VinSnippetView branch.
   - References: v3 `model_v3.py` `_sample_semidense_points` (lines ~543–583); v2 `model_v2.py` `_sample_semidense_points` (lines ~840–852).

3) **Docstring vs. channel list mismatch in v3**
   - v3 module docstring lists scene-field channels without `unknown`, but `FIELD_CHANNELS_V3` includes it by default. This is a minor documentation drift that can confuse ablation assumptions.
   - References: `model_v3.py` docstring (lines ~12–18) vs. `FIELD_CHANNELS_V3` (lines ~93–101).

## Behavior deltas vs v2 (given config)
- v2 config enables trajectory encoder, semidense frustum MHCA, and voxel-valid-fraction features. v3 removes traj/frustum/point encoders and only uses semidense stats via FiLM; this is a large capacity/conditioning change that can impact ranking quality.
- v2 concatenates semidense projection features into the head input; v3 only modulates global features via FiLM, which changes how strongly semidense evidence can steer scoring when voxel evidence is weak.
- v2 uses `use_voxel_valid_frac_gate=false` but appends `(voxel_valid_frac, 1-voxel_valid_frac)`; v3 defaults to gating and does not expose a “valid-frac feature” concat.
- v2 semidense visibility fraction is an unweighted valid/finite ratio; v3 uses weighted visibility (obs_count + inv_dist_std). This changes the meaning of `semidense_candidate_vis_frac` and affects diagnostics/thresholds.
- Default dimensions differ (v2: field_dim=48, head_hidden_dim=128, head_num_layers=2, dropout ~0.31, pool grid=5 vs v3 defaults 16/192/1/0.05/6).

## Suggestions / follow-ups
- Decide on a single CW90 convention for poses and `p3d_cameras` and enforce it in both v2 and v3; consider asserting a frame consistency flag in the batch if you want to keep it implicit.
- If `VinSnippetView` points can include NaN padding, add finite filtering to v3’s VinSnippetView sampling (match v2 behavior).
- Align the v3 docstring’s scene-field channel list with `FIELD_CHANNELS_V3`.
- If you need v3 to approximate v2 behavior under the provided config, add optional frustum MHCA + trajectory features or explicitly document the feature drop.

## Open questions
- Are `p3d_cameras` already CW90-corrected upstream in the labeler? If yes, v3’s pose undo is wrong; if not, projections are currently mismatched.
- Does the VinSnippetView cache always provide valid `lengths` and no NaNs? If not, v3’s sampling is risky.

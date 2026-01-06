#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

= Ablation Plan and Open Experiments

We prioritize ablations that test whether candidate-dependent signals improve
ordinal separation and reduce prediction collapse. Planned experiments and
expected outcomes are summarized in @tab:ablations.

#figure(
  kind: "table",
  supplement: [Table],
  placement: none,
  caption: [Planned ablations and hypotheses.],
  text(size: 8.5pt)[
    #table(
      columns: (9em, 8em, auto),
      align: (left, left, left),
      toprule(),
      table.header([Ablation], [Change], [Hypothesis]),
      midrule(),
      [Semidense point encoder],
      [`use_point_encoder` True/False (PointNeXt)],
      [Does a global semidense embedding help beyond view-conditioned cues?],
      [Semidense frustum MHCA],
      [`enable_semidense_frustum` True/False],
      [Does token-level candidate conditioning improve ranking vs. projection stats alone?],
      [Visibility token embedding],
      [`semidense_visibility_embed` True/False],
      [Token-type embedding should help keep invalid points informative without masking.],
      [Mask invalid tokens],
      [`semidense_frustum_mask_invalid` True/False],
      [Masking may stabilize attention but can erase evidence about *missing* visibility.],
      [Observation counts],
      [`semidense_include_obs_count` True/False + norm choice],
      [Track-length features should correlate with point reliability and improve frustum aggregation.],
      [Trajectory context],
      [`use_traj_encoder` True/False],
      [History-aware features should reduce ambiguity between candidates with similar geometry.],
      [Voxel reliability gating],
      [`use_voxel_valid_frac_gate` True/False],
      [Gating should reduce noise when candidates fall outside EVL voxel extent.],
      [Voxel reliability feature],
      [`use_voxel_valid_frac_feature` True/False],
      [Explicitly conditioning the head on evidence quality should improve calibration.],
      [Global pool resolution],
      [`global_pool_grid_size` in {4..8}],
      [Higher resolution may help in clutter but risks overfitting / compute overhead.],
      [CORAL imbalance handling],
      [`coral_loss_variant` ∈ {coral, balanced_bce, focal}],
      [Better gradients for rare thresholds; less median-bin collapse.],
      [Coverage-weight curriculum],
      [`coverage_weight_*` schedule on/off],
      [Early emphasis on high-evidence candidates should prevent collapse while still learning all candidates.],
      bottomrule(),
    )
  ],
) <tab:ablations>

We will report per-epoch Spearman correlation, confusion matrices, and
calibration curves for each ablation, and use these results to guide the
transition to entity-aware NBV.

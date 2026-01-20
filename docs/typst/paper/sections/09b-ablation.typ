#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

= Ablation Plan and Open Experiments

We prioritize ablations that test whether candidate-dependent signals improve
ordinal separation and reduce prediction collapse. Planned experiments and
expected outcomes are summarized in @tab:ablations.

We structure ablations as feature-contract deltas on top of the core EVL+pose
baseline: (i) add candidate validity/evidence scalars, (ii) add semi-dense
projection statistics, (iii) add token-level frustum aggregation, and (iv) add
optional trajectory and point-set encoders. This ordering makes it explicit
which additional signals are responsible for gains or failure modes.

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
      [Semi-dense point encoder],
      [on/off (PointNeXt)],
      [Does a global semi-dense embedding help beyond view-conditioned cues?],
      [Semi-dense frustum MHCA],
      [on/off],
      [Does token-level candidate conditioning improve ranking vs. projection stats alone?],
      [Visibility token embedding],
      [on/off],
      [Token-type embedding should help keep invalid points informative without masking.],
      [Mask invalid tokens],
      [on/off],
      [Masking may stabilize attention but can erase evidence about *missing* visibility.],
      [Observation counts],
      [on/off + normalization choice],
      [Track-length features should correlate with point reliability and improve frustum aggregation.],
      [Trajectory context],
      [on/off],
      [History-aware features should reduce ambiguity between candidates with similar geometry.],
      [Voxel reliability gating],
      [on/off],
      [Gating should reduce noise when candidates fall outside EVL voxel extent.],
      [Voxel reliability feature],
      [on/off],
      [Explicitly conditioning the head on evidence quality should improve calibration.],
      [Global pool resolution],
      [grid size $G$ in {$4, 5, 6, 7, 8$}],
      [Higher resolution may help in clutter but risks overfitting / compute overhead.],
      [CORAL imbalance handling],
      [loss variant: CORAL vs. balanced vs. focal],
      [Better gradients for rare thresholds; less median-bin collapse.],
      [Coverage-weight curriculum],
      [on/off (annealed weighting)],
      [Early emphasis on high-evidence candidates should prevent collapse while still learning all candidates.],
      bottomrule(),
    )
  ],
) <tab:ablations>

We will report per-epoch Spearman correlation, confusion matrices, and
calibration curves for each ablation, and use these results to guide the
transition to entity-aware NBV.

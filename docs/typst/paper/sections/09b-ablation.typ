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
      [Frustum MHCA],
      [Remove frustum tokens],
      [Less view-conditioned evidence → higher collapse risk.],
      [Semidense projection],
      [Disable projection stats],
      [Reduced robustness when voxel context is out-of-bounds.],
      [PointNeXt embedding],
      [Freeze vs. drop],
      [Test whether global geometry improves ranking beyond view cues.],
      [Balanced CORAL loss],
      [Coral vs. focal],
      [Improved gradients for rare thresholds; less median-bin collapse.],
      [Stage feature],
      [Add coverage proxy],
      [Improve calibration across early/late acquisition stages.],
      bottomrule(),
    )
  ],
) <tab:ablations>

We will report per-epoch Spearman correlation, confusion matrices, and
calibration curves for each ablation, and use these results to guide the
transition to entity-aware NBV.

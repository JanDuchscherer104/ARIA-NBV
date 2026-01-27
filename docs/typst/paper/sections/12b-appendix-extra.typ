#pagebreak()

= Appendix: Additional Diagnostics

#import "../../shared/macros.typ": *
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

This appendix page contains additional distribution plots used during candidate
sampling verification.

#figure(
  image("/figures/app/view_hist_full_az_unfi_sphere_120.png", width: 100%),
  caption: [Azimuth distribution for uniform sphere sampling.]
) <fig:view-hist-unfi>

#figure(
  image("/figures/app/view_hist_full_roll_jitter_20.png", width: 100%),
  caption: [Roll jitter distribution across candidates.]
) <fig:roll-jitter>

== Logged training metrics (VIN Lightning)

We log loss scalars and diagnostic metrics with keys of the form
#code-inline[stage/metric], where #code-inline[stage] is one of
#code-inline[train], #code-inline[val], or #code-inline[test]. Step-level
metrics use the suffix #code-inline[step] with a leading underscore and are
emitted only during training.

#figure(
  kind: "table",
  supplement: [Table],
  caption: [Loss scalars logged by the VIN Lightning module.],
  text(size: 8.5pt)[
    #table(
      columns: (12em, auto),
      align: (left, left),
      toprule(),
      table.header([Key], [Definition]),
      midrule(),
      [#code-inline[loss]], [Combined training loss (CORAL + auxiliary regression).],
      [#code-inline[coral_loss]], [Mean CORAL ordinal loss.],
      [#code-inline[coral_loss_rel_random]], [CORAL loss normalized by a random baseline.],
      [#code-inline[coral_loss_balanced_bce]], [Balanced-BCE variant of CORAL (diagnostic).],
      [#code-inline[coral_loss_focal]], [Focal variant of CORAL (diagnostic).],
      [#code-inline[aux_regression_loss]], [Auxiliary regression loss on expected RRI (Huber/MSE).],
      bottomrule(),
    )
  ],
) <tab:vin-losses>

#figure(
  kind: "table",
  supplement: [Table],
  caption: [Auxiliary and diagnostic metrics logged by the VIN Lightning module.],
  text(size: 8.5pt)[
    #table(
      columns: (14em, auto),
      align: (left, left),
      toprule(),
      table.header([Key], [Definition]),
      midrule(),
      [#code-inline[rri_mean]], [Mean oracle RRI over valid candidates.],
      [#code-inline[pred_rri_mean]], [Mean predicted RRI proxy (expected CORAL value).],
      [#code-inline[pred_rri_bias2]], [Bias squared of prediction error (val only).],
      [#code-inline[pred_rri_variance]], [Variance of prediction error (val only).],
      [#code-inline[top3_accuracy]], [Top-3 classification accuracy from ordinal probabilities.],
      [#code-inline[aux_regression_weight]], [Current auxiliary-loss weight after decay.],
      [#code-inline[coral_monotonicity_violation_rate]], [Fraction of logits violating CORAL monotonicity.],
      [#code-inline[voxel_valid_frac_mean]], [Mean voxel coverage proxy $v_i$.],
      [#code-inline[voxel_valid_frac_std]], [Std of voxel coverage proxy.],
      [#code-inline[semidense_candidate_vis_frac_mean]], [Mean semi-dense visibility fraction $v_i^("sem")$.],
      [#code-inline[semidense_candidate_vis_frac_std]], [Std of semi-dense visibility fraction.],
      [#code-inline[semidense_valid_frac_mean]], [Alias of semi-dense visibility fraction (legacy).],
      [#code-inline[semidense_valid_frac_std]], [Std of legacy semi-dense visibility fraction.],
      [#code-inline[candidate_valid_frac]], [Fraction of candidates passing validity checks.],
      [#code-inline[coverage_weight_mean]], [Mean coverage-based loss weight (train only).],
      [#code-inline[coverage_weight_strength]], [Current coverage-weight strength (train only).],
      [#code-inline[spearman]], [Spearman correlation between predicted scores and RRI.],
      [#code-inline[spearman_step]], [Step-level Spearman (train interval).],
      [#code-inline[confusion_matrix]], [Ordinal confusion matrix (epoch).],
      [#code-inline[confusion_matrix_step]], [Step-level confusion matrix (train interval).],
      [#code-inline[label_histogram]], [Histogram of ordinal labels (epoch).],
      [#code-inline[label_histogram_step]], [Step-level label histogram (train interval).],
      bottomrule(),
    )
  ],
) <tab:vin-metrics>

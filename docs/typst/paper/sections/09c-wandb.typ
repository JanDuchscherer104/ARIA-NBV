#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

= WandB Run Analysis (Jan 3, 2026)

We reviewed all WandB runs started on January 3, 2026 with more than 500 logged
steps (based on the `wandb-summary.json` files). Three runs met this criterion.
Their key metrics are summarized in @tab:wandb-runs.

#figure(
  kind: "table",
  supplement: [Table],
  placement: none,
  caption: [Summary of January 3, 2026 WandB runs with >500 steps.],
  text(size: 8.5pt)[
    #table(
      columns: (8em, auto),
      align: (left, left),
      toprule(),
      table.header([Run ID], [Summary]),
      midrule(),
      [nn0jcqwr],
      [Start 01:02:53; global_step 0 (log step 1341). Loss NaN from start; optimizer loop likely never advanced.],
      [m06auwmr],
      [Start 01:14:45; global_step 547 (log step 1604). Train/val losses NaN; val spearman 0.004; top3 0.191.],
      [jejo31ut],
      [Start 01:45:43; global_step 1506 (log step 4425). Finite losses (train 10.15, val 7.69); val spearman 0.127; top3 0.222.],
      bottomrule(),
    )
  ],
) <tab:wandb-runs>

== Observations

- Two runs (nn0jcqwr, m06auwmr) reported NaN losses, indicating numerical
  instability or invalid labels early in training. In nn0jcqwr, `global_step`
  stayed at 0 despite >1000 logged steps, suggesting a failure before the
  optimizer update loop.
- The most recent run (jejo31ut) produced stable losses and non-zero validation
  correlation (Spearman 0.127) with a top-3 accuracy of 0.222. Balanced BCE and
  focal CORAL losses were logged alongside the primary loss, showing that the
  balanced variants remain well-behaved even when the main CORAL loss is large.
- Monotonicity violation rates were 0 across runs, indicating that the ordinal
  thresholds remained properly ordered when the loss was finite.

These results support the emphasis on stronger candidate-specific signals
(frustum MHCA, semidense projection features) and careful loss balancing to
avoid collapse.

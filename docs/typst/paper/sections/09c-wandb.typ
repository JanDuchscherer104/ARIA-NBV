#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

= WandB Run Analysis (Jan 3, 2026)

We reviewed all WandB runs started on January 3, 2026 with more than 500 logged
steps (based on the run summary metadata). Three runs met this criterion. Their
key metrics are summarized in @tab:wandb-runs.

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
      table.header([Run], [Summary]),
      midrule(),
      [Run 1],
      [Loss NaN from the start; optimizer loop likely never advanced.],
      [Run 2],
      [Train/val losses NaN; validation Spearman 0.004; top-3 bin accuracy 0.191.],
      [Run 3],
      [Finite losses (train 10.15, val 7.69); validation Spearman 0.127; top-3 bin accuracy 0.222.],
      bottomrule(),
    )
  ],
) <tab:wandb-runs>

== Observations

- Two runs reported NaN losses, indicating numerical instability or invalid
  labels early in training. In one case, the optimizer update loop likely never
  executed.
- The stable run produced non-zero validation correlation (Spearman 0.127) and
  a top-3 bin accuracy of 0.222. Balanced and focal threshold losses were logged
  alongside the primary CORAL loss, suggesting that the imbalance-aware
  variants remain well-behaved even when the main CORAL loss is large.
- Monotonicity violation rates were 0 across runs, indicating that the ordinal
  thresholds remained properly ordered when the loss was finite.

These results support the emphasis on stronger candidate-specific signals
(frustum MHCA, semi-dense projection features) and careful loss balancing to
avoid collapse.

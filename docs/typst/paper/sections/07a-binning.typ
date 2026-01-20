= Stage-Aware Binning and Priors

Oracle RRI distributions shift over the course of a trajectory. Early
acquisition stages tend to yield larger improvements, while late stages produce
smaller but still valuable refinements. VIN-NBV addresses this effect with
stage-normalized targets @VIN-NBV-frahm2025. Our current baseline uses a single
global quantile binner fitted on oracle RRIs; stage-aware normalization is left
as future work and is summarized here for completeness.

== Stage normalization

A common normalization uses a per-stage z-score followed by a tanh clip before
binning. Given stage group $s$ with mean $mu_s$ and standard deviation
$sigma_s$,

#block[
  #align(center)[
    $ r'_s = "tanh"((r - mu_s) / sigma_s) $
  ]
]

where $r$ is the raw RRI. This reduces variance across acquisition stages and
produces more stable ordinal targets.
We do not apply stage normalization in the current baseline; instead we fit a
single global binner and expose stage-aware binning as a planned ablation.

Stage-dependent examples are provided in the appendix (early vs. late frames).

== Prior-aligned initialization

CORAL thresholds are imbalanced by construction. We therefore initialize bias
terms from the empirical cumulative priors estimated by the fitted binner.
This reduces early collapse toward a constant predictor and aligns the model
with the dataset's base rate of improvement.

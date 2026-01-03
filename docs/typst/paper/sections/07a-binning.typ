= Stage-Aware Binning and Priors

RRI distributions shift over the course of a trajectory. Early acquisition
stages tend to yield larger improvements, while late stages produce smaller but
still valuable refinements. VIN-NBV addresses this by stage-normalizing the RRI
values before binning @VIN-NBV-frahm2025. We follow the same idea and plan to
track stage index (or coverage proxies) as additional model inputs.

== Stage normalization

A simple normalization uses a per-stage z-score followed by a tanh clip before
binning. Given stage group $s$ with mean $mu_s$ and standard deviation $sigma_s$,

#block[
  #align(center)[
    $ r'_s = "tanh"((r - mu_s) / sigma_s) $
  ]
]

where $r$ is the raw RRI. This reduces variance across acquisition stages and
produces more stable ordinal targets.

Stage-dependent examples are provided in the appendix (early vs. late frames).

== Prior-aligned initialization

CORAL thresholds are imbalanced by construction. We therefore initialize bias
terms from the empirical cumulative priors estimated by the fitted binner.
This reduces early collapse toward a constant predictor and aligns the model
with the dataset's base rate of improvement.

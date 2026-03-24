= Training Objective

#import "../../shared/macros.typ": *

// TODO(paper-cleanup): Keep notation consistent with slides/macros:
// - use `RRI` macro consistently in text,
// - use `bold(...)` for vector-valued quantities (e.g., logits/probabilities),
// - prefer referencing `#eqs` definitions instead of reintroducing symbols ad-hoc.

// <rm>
// Scaffolding + typo (“training a our”). Replace with a single direct statement like
// “We train VINv3 with an ordinal regression objective on oracle RRI labels.”
To enable training a our candidate scorer on oracle RRI labels,
we describe an ordinal regression objective.
// </rm>
RRI values are binned into $K$
ordered classes. CORAL models the cumulative probability that the class exceeds
each threshold, yielding $K-1$ logits $ell_k$ (collectively
$bold(ell) in bb(R)^(K-1)$) and cumulative probabilities
$bold(p) = sigma(bold(ell))$ @CORAL-cao2019. These are *cumulative*
probabilities, not class marginals.

== CORAL loss

Let $r$ be the continuous RRI and $y$ the ordinal bin index. The CORAL targets
are binary levels $t_k = bb(1)[y > k]$. The per-sample loss is

#text(size: 8.5pt)[#eqs.coral.loss]

To recover a scalar prediction, we convert cumulative probabilities into
marginal class probabilities:

#block[#align(center)[#eqs.coral.marginals]]

The expected RRI is then

#block[#align(center)[#eqs.coral.expected]]

where $u_k$ is a representative value for bin $k$ (initialized from bin means
and optionally learned). This matches the CORAL paper's ordinal semantics and
avoids treating cumulative probabilities as class posteriors.

When $u_k$ is learned, we constrain it to be monotone (e.g., via a cumulative
softplus parameterization) to preserve ordinal ordering.

We additionally monitor whether CORAL's cumulative probabilities are rank
consistent. Since $p_k = P(y > k)$ should be non-increasing in $k$, we define a
monotonicity violation rate to monitor the correctness of our modifications to the CORAL setup:

#block[#align(center)[#eqs.coral.violation]]

and report its mean over valid candidates as a diagnostic.

// <rm>
// Too detailed for main text; move to appendix or repo docs and keep only equations + the final
// objective here.
== Implementation deltas vs. coral-pytorch

We build on the reference implementation in `coral-pytorch` @coral-pytorch-2025
and extend it for RRI regression:

- *Label handling*: convert ordinal labels to level targets internally
  (no external `levels_from_labelbatch` step).
- *Class marginals*: derive $pi_k$ from cumulative probabilities, then compute
  expected RRI with bin representatives (Eq. above).
- *Learnable bin values*: treat $u_k$ as monotone learnable scalars via
  softplus deltas, // TODO: note that u_0 and delta_j are learnable parameters and intialized from bin means.
  #block[#align(center)[#eqs.coral.bin_values]]
- *Bias initialization from priors*: optional initialization of per-threshold
  biases from fitted class priors (instead of fixed descending bias values).
- *Loss variants*: support balanced BCE and focal threshold losses to mitigate
  imbalance (definitions in Appendix @sec:appendix-extra).
- *Diagnostics*: log monotonicity violations and a relative-to-random baseline
  (#eqs.coral.rel_random) for calibration tracking.
// </rm>

== Auxiliary regression

We optionally add a Huber loss on $hat(r)$ to stabilize early training and
improve calibration. The final objective is

#block[#align(center)[#eqs.vin.loss_total]]

The auxiliary weight $lambda$ is decayed over training to encourage sharper
ordinal separation while retaining a meaningful continuous prediction. In the
current training config we use $lambda_0 = 10$, $gamma = 0.99$,
$lambda_"min" = 0.1$, and apply the decay once per epoch.
// <rm>
// Run/config-specific hyperparameters in prose; source from a config artifact or move to a
// baseline spec table.
// </rm>
// TODO(paper-cleanup): These hyperparameters are easy to drift; source them from the same
// config artifact as the training table / slides (avoid hard-coding in prose).

== Threshold imbalance and balancing

Ordinal thresholds are inherently imbalanced because early thresholds have high
positive rates. To mitigate collapse toward constant predictions, we support
balanced BCE or focal variants over thresholds, using priors from the fitted
binner. This retains CORAL semantics while improving gradients for rare
thresholds @FocalLoss-lin2017. We also log monotonicity violations to ensure
ordinal consistency.

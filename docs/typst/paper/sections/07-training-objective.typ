= Training Objective

#import "/typst/shared/macros.typ": *

To enable future learning of a VIN-style candidate scorer on oracle RRI labels,
we describe an ordinal regression objective. RRI values are binned into $K$
ordered classes. CORAL models the cumulative probability that the class exceeds
each threshold, yielding $K-1$ logits $ell_k$ (collectively
$bold(ell) in bb(R)^(K-1)$) and cumulative probabilities
$bold(p) = sigma(bold(ell))$ @CORAL-cao2019. These are *cumulative*
probabilities, not class marginals.

== CORAL loss

Let $r$ be the continuous RRI and $y$ the ordinal bin index. The CORAL targets
are binary levels $t_k = bb(1)[y > k]$. The per-sample loss is

#block[
  #align(center)[
    $
      #sym_loss _("coral")(y, bold(p))
      = - sum_(k=0)^(K-2) (t_k "log"(p_k) + (1 - t_k) "log"(1 - p_k))
    $
  ]
]

To recover a scalar prediction, we convert cumulative probabilities into
marginal class probabilities:

#block[
  #align(center)[
    $ pi_k = p_(k-1) - p_k, quad p_(-1) = 1, quad p_(K-1) = 0 $
  ]
]

The expected RRI is then

#block[
  #align(center)[
    $ hat(r) = sum_(k=0)^(K-1) pi_k dot u_k $
  ]
]

where $u_k$ is a representative value for bin $k$ (initialized from bin means
and optionally learned). This matches the CORAL paper's ordinal semantics and
avoids treating cumulative probabilities as class posteriors.

When $u_k$ is learned, we constrain it to be monotone (e.g., via a cumulative
softplus parameterization) to preserve ordinal ordering.

We additionally monitor whether CORAL's cumulative probabilities are rank
consistent. Since $p_k = P(y > k)$ should be non-increasing in $k$, we define a
monotonicity violation rate

#block[
  #align(center)[
    $
      v =
      (1) / (K - 2)
      sum_(k=0)^(K-3) bb(1)[p_(k+1) > p_k]
    $
  ]
]

and report its mean over valid candidates as a diagnostic.

== Auxiliary regression

We optionally add a Huber loss on $hat(r)$ to stabilize early training and
improve calibration. The final objective is

#block[
  #align(center)[
    $ #sym_loss = #sym_loss _"coral" + lambda dot #sym_loss _"reg" $
  ]
]

The auxiliary weight $lambda$ is decayed over training to encourage sharper
ordinal separation while retaining a meaningful continuous prediction.

== Threshold imbalance and balancing

Ordinal thresholds are inherently imbalanced because early thresholds have high
positive rates. To mitigate collapse toward constant predictions, we support
balanced BCE or focal variants over thresholds, using priors from the fitted
binner. This retains CORAL semantics while improving gradients for rare
thresholds @FocalLoss-lin2017. We also log monotonicity violations to ensure
ordinal consistency.

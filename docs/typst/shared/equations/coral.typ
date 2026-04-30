#import "../symbols.typ": symb

#let coral = (
    loss: $
      cal(L)_"coral" (y, bold(p))
      = - sum_(k=0)^(K-2) (t_k "log"(p_k) + (1 - t_k) "log"(1 - p_k))
    $,
    balanced_bce: $
      cal(L)_"bal"
      = -(1)/(K-1) sum_(k=0)^(K-2)
      (w_k t_k "log"(p_k) + (1 - t_k) "log"(1 - p_k))
    $,
    balanced_bce_weight: $ w_k = (1 - pi_k^("th")) / pi_k^("th") $,
    focal: $
      cal(L)_"focal"
      = -(1)/(K-1) sum_(k=0)^(K-2)
      alpha_(t,k) (1 - p_(t,k))^gamma "log"(p_(t,k))
    $,
    focal_defs: $
      p_(t,k) = p_k t_k + (1 - p_k) (1 - t_k),
      quad alpha_(t,k) = alpha t_k + (1 - alpha) (1 - t_k)
    $,
    marginals: $ pi_k = p_(k-1) - p_k, quad p_(-1) = 1, quad p_(K-1) = 0 $,
    expected: $ hat(r) = sum_(k=0)^(K-1) pi_k dot u_k $,
    bin_values: $
      u_0 in bb(R),
      quad u_k = u_0 + sum_(j=1)^k op("softplus")(delta_j)
    $,
    // Fraction of rank-order violations in the cumulative probabilities.
    violation: $
      v =
      (1) / (K - 2)
      sum_(k=0)^(K-3) bb(1)[p_(k+1) > p_k]
    $,
    rel_random: $ cal(L)_("rel") = cal(L)_("coral") / ((K - 1) "log"(2)) $,
  )

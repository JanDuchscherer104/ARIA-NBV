#import "../symbols.typ": symb

#let coverage = (
    ratio: $ "CR"_t = (tilde(N)_t) / (N^*) dot 100% $,
    weight: $ w_i(t) = (1 - lambda_t) + lambda_t (f + (1 - f) c_i^p) $,
    weighted_loss: $ cal(L) = (sum_i w_i(t) ell_i) / (sum_i w_i(t)) $,
    strength_linear: $
      lambda_t = lambda_0 + (lambda_T - lambda_0) dot (t / T)
    $,
    strength_cosine: $
      lambda_t = lambda_T + (lambda_0 - lambda_T) dot (1 + "cos"(pi t / T)) / 2
    $,
  )

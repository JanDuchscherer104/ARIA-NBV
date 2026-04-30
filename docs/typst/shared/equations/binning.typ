#import "../symbols.typ": symb

#let binning = (
    // Empirical quantile edges (equal-mass bins) for discretizing oracle RRI.
    edges: $
      e_k = "Quantile"( {r_i}_(i=1)^N, k/K),
      quad k in {1, dots, K-1}
    $,
    // Ordinal class index via edge counting (equivalent to `torch.bucketize`).
    label: $
      y(r) = sum_(k=1)^(K-1) bb(1)[r > e_k],
      quad y(r) in {0, dots, K-1}
    $,
    // CORAL level targets derived from ordinal labels.
    levels: $
      t_k = bb(1)[y > k],
      quad k in {0, dots, K-2}
    $,
  )

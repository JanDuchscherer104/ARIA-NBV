

= Evaluation Protocol: Metrics

- Bidirectional surface error: overall reconstruction quality (accuracy + completeness).
- Accuracy and completeness: directional components for diagnosis.
- Oracle RRI: normalized improvement per candidate.
- Ranking and ordinal metrics: Spearman correlation, top-k bin accuracy,
  confusion matrices, and label histograms for RRI bins.

When training a candidate scorer, these metrics are logged per epoch and per
interval to detect collapse and to validate the effect of imbalance-aware
ordinal losses. In this paper, we provide the definitions and use them for
oracle-label diagnostics.

== Ranking and ordinal metrics

Let $s_i in [0, 1]$ be the predicted score for candidate $i$ (the normalized
expected CORAL bin) and $r_i$ the oracle RRI. We measure ranking agreement with
Spearman's rank correlation @Spearman1904:
// TODO(paper-cleanup): Align symbols with macros (`#(symb.vin.rri_hat)_i`, `#(symb.vin.rri)_i`)
// and reuse `#eqs.metrics.spearman` for the definition.

#block[
  #align(center)[
    $
      rho =
      op("corr")(
        op("rank") {s_i}_(i in cal(I)),
        op("rank") {r_i}_(i in cal(I))
      )
    $
  ]
]

// <rm>
// “Current runs …” + hard-coded split counts read like internal bookkeeping. Prefer a single
// reproducibility table (scene-disjoint split + seed) and an NBV rollout evaluation section.
Current runs report all metrics defined in `VinLightningModule`: CORAL losses,
auxiliary regression diagnostics, Spearman correlation, top-3 accuracy,
confusion matrices, label histograms, monotonicity violations, and coverage
fractions (voxel and semidense). Results in this paper use the ASE-EFM GT split
from the offline cache (80 scenes, 883 snippets; train/val split 706/177).
// </rm>
// TODO(paper-cleanup): Replace these numbers with imported cache stats (and ensure they match
// figures/tables elsewhere). Avoid repeating them across sections.

where $cal(I)$ contains candidates with finite oracle RRI and finite model
outputs. We compute $rho$ over all valid candidates accumulated across an epoch
(flattened over snippets and candidates).

For ordinal supervision, let $y_i in {0, dots, K - 1}$ be the binned oracle label
and $bold(p)_i in [0, 1]^K$ the predicted class probabilities. We report top-k
bin accuracy
// TODO(paper-cleanup): Reuse `#eqs.metrics.topk_acc` and define `bold(p)_i` precisely:
// CORAL yields cumulative probs `p_k = P(y > k)`, so `bold(pi)` (marginals) may be the
// more appropriate input to TopK depending on how it is computed in code.

#block[
  #align(center)[
    $
      "Acc@k" =
      (1) / (|cal(I)|)
      sum_(i in cal(I)) bb(1)[y_i in op("TopK")(bold(p)_i, k)].
    $
  ]
]

Here, $op("TopK")(bold(p)_i, k)$ denotes the set of the $k$ most probable bins
under $bold(p)_i$.

In our logs we use $k = 3$. Confusion matrices and label histograms are computed
from $(hat(y)_i, y_i)$, where $hat(y)_i$ is the decoded ordinal class from CORAL
logits, to expose class imbalance and mode collapse:


#block[
  #align(center)[
    $
      C_(a,b) = |{i in cal(I) | hat(y)_i = a, y_i = b}| \
      h_b = |{i in cal(I) | y_i = b}|.
    $
  ]
]

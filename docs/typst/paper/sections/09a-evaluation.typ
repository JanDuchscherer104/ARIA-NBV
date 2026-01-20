= Evaluation Protocol

We evaluate oracle RRI (and, in future, model predictions) using surface reconstruction
metrics derived from point↔mesh distances. Following the standard
accuracy/completeness decomposition, we measure:

- *Accuracy (prediction→GT)*: mean squared point-to-triangle distance from the
  reconstruction point set to the ground-truth mesh.
- *Completeness (GT→prediction)*: mean squared triangle-to-point distance from
  the ground-truth mesh to the reconstruction point set.

We define a bidirectional Chamfer-style surface error as the sum of the two
components and compute oracle RRI as the relative reduction in this error after
adding a candidate view (see @sec:problem). We adopt ATEK's terminology and reporting
conventions for the directional components @ATEK-SurfaceRecon-2025.

#figure(
  image("/figures/atek/overview.png", width: 100%),
  caption: [ATEK surface reconstruction evaluation pipeline overview.]
) <fig:atek-overview>

== Metrics reported

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

where $cal(I)$ contains candidates with finite oracle RRI and finite model
outputs. We compute $rho$ over all valid candidates accumulated across an epoch
(flattened over snippets and candidates).

For ordinal supervision, let $y_i in {0, dots, K - 1}$ be the binned oracle label
and $bold(p)_i in [0, 1]^K$ the predicted class probabilities. We report top-k
bin accuracy

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

Top-k accuracy is a standard multi-class evaluation primitive (e.g. the ImageNet
challenge) @ILSVRC-russakovsky2014.

In our logs we use $k = 3$. Confusion matrices and label histograms are computed
from $(hat(y)_i, y_i)$, where $hat(y)_i$ is the decoded ordinal class from CORAL
logits, to expose class imbalance and mode collapse:

#block[
  #align(center)[
    $
      C_(a,b) = |{i in cal(I) | hat(y)_i = a, y_i = b}|,
      quad
      h_b = |{i in cal(I) | y_i = b}|.
    $
  ]
]

= Evaluation Protocol

We evaluate oracle RRI using standard surface reconstruction metrics derived
from point-to-mesh distances. The symmetric Chamfer distance decomposes into
accuracy (prediction to GT) and completeness (GT to prediction) terms, which we
log separately to diagnose whether a candidate improves coverage or reduces
outlier errors. We also compute precision, recall, and F-score at fixed
thresholds (typically 5 cm) following ATEK's surface reconstruction protocol
@ATEK-SurfaceRecon-2025.

#figure(
  image("/figures/atek/overview.png", width: 100%),
  caption: [ATEK surface reconstruction evaluation pipeline overview.]
) <fig:atek-overview>

== Metrics reported

- Chamfer distance (symmetric): overall reconstruction quality.
- Accuracy and completeness: directional components of Chamfer.
- Precision and recall at threshold: sensitivity to surface outliers.
- Ordinal metrics: confusion matrices and label histograms for RRI bins.

These metrics are logged per epoch and per interval to detect training collapse
and to validate the effect of balanced threshold losses.

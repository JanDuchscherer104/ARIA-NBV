= Diagnostics and Preliminary Findings

We instrument the oracle label pipeline with extensive diagnostics to
understand failure modes and to validate geometric conventions. The dashboard
visualizes candidate distributions, rendered depth maps, and point↔mesh surface
errors, and reports oracle RRI distributions. For future learning-based
experiments, we also define ranking and ordinal metrics (Spearman correlation,
top-k bin accuracy, confusion matrices) that diagnose collapse and calibration.

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 8pt,
    image("/figures/app/dir_dist_full_circ.png", width: 100%),
    image("/figures/app/candidates_full_circ.png", width: 100%),
    image("/figures/app/collision.png", width: 100%),
  ),
  caption: [Streamlit diagnostics for candidate sampling: direction/distance coverage (left), candidate distribution in the reference frame (center), and collision checking against the mesh (right).],
) <fig:diagnostics-candidates>

Additional diagnostic figures (candidate frusta, depth histograms, semi-dense
overlays) are provided in the appendix for completeness.

== Failure modes

Early experiments reveal two recurring issues. First, when candidate-specific
signals are weak, the model can collapse to a narrow ordinal prediction band.
Second, EVL voxel features may be out-of-bounds for wide candidate shells,
leading to low-quality global context. These observations motivated the
semi-dense projection features and frustum-aware attention used in the current
architecture.

== Monitoring and evaluation

We log confusion matrices and label histograms per epoch, and compute
monotonicity violation rates of CORAL thresholds to detect ordinal instability.
These diagnostics are critical for verifying that the ordinal loss, auxiliary
regression, and threshold balancing behave as intended.

= Diagnostics and Preliminary Findings

We instrument the training and oracle pipeline with extensive diagnostics to
understand failure modes and guide architecture decisions. The metrics include
Spearman correlation with oracle RRI, confusion matrices for ordinal bins, and
per-module gradient norms. Visualization panels expose candidate distributions,
rendered depth maps, and intermediate VIN features.

Key diagnostic figures (candidate frusta, depth histograms, semidense overlays)
are moved to the appendix to reduce clutter in the main text.

== Failure modes

Early experiments reveal two recurring issues. First, when candidate-specific
signals are weak, the model can collapse to a narrow ordinal prediction band.
Second, EVL voxel features may be out-of-bounds for wide candidate shells,
leading to low-quality global context. These observations motivated the
semidense projection features and frustum-aware attention used in the current
architecture.

== Monitoring and evaluation

We log confusion matrices and label histograms per epoch, and compute
monotonicity violation rates of CORAL thresholds to detect ordinal instability.
These diagnostics are critical for verifying that the ordinal loss, auxiliary
regression, and threshold balancing behave as intended.

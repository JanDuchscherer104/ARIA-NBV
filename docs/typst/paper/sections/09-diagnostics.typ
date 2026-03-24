= Diagnostics and Preliminary Findings

We instrument the oracle label pipeline with extensive diagnostics to
understand failure modes and to validate geometric conventions. The dashboard
visualizes candidate distributions, rendered depth maps, and point↔mesh surface
errors, and reports oracle RRI distributions. For future learning-based
experiments, we also define ranking and ordinal metrics (Spearman correlation,
top-k bin accuracy, confusion matrices) that diagnose collapse and calibration.

Additional diagnostic figures (candidate sampling, depth histograms, and
semi-dense overlays) are shown throughout the implementation sections:
@fig:candidate-poses and @fig:streamlit-diagnostics (pipeline), and
@fig:oracle-depth-renders, @fig:oracle-depth-histograms, and
@fig:oracle-fusion-diagnostics (oracle RRI).

== Failure modes

Early experiments reveal two recurring issues. First, when candidate-specific
signals are weak, the model can collapse to a narrow ordinal prediction band.
Second, EVL voxel features may be out-of-bounds for wide candidate shells,
leading to low-quality global context. These observations motivated the
semi-dense projection features in the current VIN v3 baseline and, in VIN v2
ablations, optional frustum-aware attention.

*Failure → fix example (frame mismatch).* When `rotate_yaw_cw90` was applied to
candidate poses but not to the PyTorch3D cameras, semidense projection features
were computed in a different camera frame than the pose encoder. This produced
inconsistent validity masks and degraded training. We addressed this by
requiring `p3d_cameras.cw90_corrected=True` whenever
`apply_cw90_correction=True` and by disabling display-only rotations in model
inputs (see Appendix @sec:appendix-pose-frames).

== Monitoring and evaluation

We log confusion matrices and label histograms per epoch, and compute
monotonicity violation rates of CORAL thresholds to detect ordinal instability.
These diagnostics are critical for verifying that the ordinal loss, auxiliary
regression, and threshold balancing behave as intended.

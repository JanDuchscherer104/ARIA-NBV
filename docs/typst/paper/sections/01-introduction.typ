= Introduction

// TODO(paper-cleanup): Reduce repeated VIN-NBV citations; cite broader NBV prior work for
// coverage/info-gain claims (or narrow the claim).
// TODO(paper-cleanup): Keep intro focused on *this* contribution (oracle labeling pipeline);
// move RL/GenNBV detail to Related Work unless it directly motivates design choices here.

Active 3D reconstruction systems must decide where to move the sensor next in
order to maximize reconstruction quality under limited capture budgets
@VIN-NBV-frahm2025. This next-best-view (NBV) problem is especially challenging
for egocentric platforms where scenes are large, cluttered, and partially
observed. Many classical NBV methods rely on proxy criteria
such as coverage or information gain, which can miss occluded or geometrically
complex regions @VIN-NBV-frahm2025. Recent methods such as VIN-NBV instead optimize reconstruction
quality directly by predicting Relative Reconstruction Improvement (RRI) for
candidate views in object-centric capture settings @VIN-NBV-frahm2025. Our goal
is to bring this quality-driven paradigm to the Aria ecosystem by combining the
Aria Synthetic Environments (ASE) dataset, the EFM3D/EVL foundation model stack,
and an oracle RRI computation pipeline tailored to egocentric trajectories
@EFM3D-straub2024 @ProjectAria-ASE-2025. Project Aria provides an egocentric
multi-modal recording platform and toolchain, including calibrated, time-aligned
sensor streams and machine perception services that recover accurate
trajectories and semi-dense reconstructions from raw recordings
@projectaria-engel2023. Importantly, the ecosystem spans both scalable synthetic
data (ASE) and real-world Aria datasets (e.g., AEO in the EFM3D release), which
share the same geometric primitives (trajectory, cameras, semi-dense SLAM
points) and conventions @EFM3D-straub2024. This makes sim-to-real transfer
directly testable: we can iterate on expensive supervision (oracle RRI) in
simulation while keeping the learned scoring model compatible with real Aria
recordings.
// TODO(paper-cleanup): “directly testable” is a strong claim; qualify scope (ASE→AEO?),
// and add a concrete eval protocol/experiment reference or soften wording.

Concurrently, reinforcement-learning approaches such as GenNBV learn continuous
5-DoF free-space policies, using multi-source state embeddings and coverage-gain
rewards to improve cross-dataset generalization @GenNBV-chen2024. However,
coverage remains a surrogate for reconstruction quality and can under-prioritize
occlusions and fine details @VIN-NBV-frahm2025. VIN-NBV instead leverages an oracle RRI signal to
train a lightweight candidate-ranking network via imitation learning, making
the objective directly reconstruction-quality aligned @VIN-NBV-frahm2025.

We do not yet learn an end-to-end action policy. Instead, we first establish a
high-fidelity supervision signal and train a model that predicts it, which can
later serve as the backbone for an NBV policy. This focus is practical: a
single candidate evaluation requires depth rendering and point#(sym.arrow.l.r)
mesh distance computations across many candidate views. Following VIN-NBV, we
discretize the action space into a candidate set and compute per-candidate
oracle RRI labels @VIN-NBV-frahm2025. We then train a VIN v3 candidate scorer on
these labels using a frozen EVL backbone and report training diagnostics
// <rm>
// Run-specific training-dynamics section; remove from main narrative if `09c-wandb.typ` is moved
// out of the paper.
(Section @sec:wandb-analysis and Appendix @sec:appendix-extra).
// </rm>
Learning a fully integrated NBV policy remains future work.
// TODO(paper-cleanup): Fix “point#(sym.arrow.l.r)mesh” formatting; use consistent notation
// (e.g., “point #sym.arrow.l.r mesh” or “point↔mesh” in text, and #symb/#eqs in math).


We target the following setting. Each ASE scene provides a prerecorded Aria
trajectory with RGB and SLAM cameras, semi-dense SLAM points, and (for a
100-scene validation subset) a watertight ground-truth mesh. We sample candidate camera poses around the
current rig pose, render candidate depth maps from the mesh, and compute oracle RRI scores by scoring the relative reconstruction improvement that the backprojected candidate depths would provide when fused with the existing semi-dense SLAM points.
// TODO(paper-cleanup): Break this long sentence; refer to Section @sec:oracle-rri for the
// pipeline and use #symb notation (e.g., $#(symb.oracle.points)_t$, #symb.ase.mesh).
// TODO(paper-cleanup): Verify “100-scene validation subset” and “watertight” wording against
// the actual ASE mesh subset definition in the EFM3D release.

*Contributions.* This paper documents an oracle supervision pipeline and a
trained VIN v3 baseline for quality-driven NBV on Aria Synthetic Environments:

1. Oracle RRI labeling pipeline for ASE that samples candidate
  views, renders metric depth from ground-truth meshes, backprojects candidate
  point clouds, and evaluates reconstruction quality against the ground-truth
  mesh.
2. Reproducible offline cache of oracle samples (candidates,
  renders/point clouds, labels, and frozen EVL features) to enable batched training without re-running the expensive oracle computations.
3. Trained VIN v3 candidate scorer on frozen EVL voxel features
  with view-conditioned evidence (pose encoding, voxel coverage proxies,
  semi-dense projection cues) and an ordinal CORAL head; we report
  training/validation diagnostics.
// <rm>
// “Framework” contribution reads like internal tooling / devlog; either compress to a single
// sentence or move to appendix/repo docs.
4. Diagnostics and convenience first research framework around PyTorch Lightning
  (modular components, declarative experiment configs, CLI utilities for data
  download/cache building/training/analysis, logging + experiment
  tracking + HParam sweeps) to
  inspect candidate validity/coverage, projection maps, loss/metric behavior and other relevant signals during development.
// </rm>
// TODO(paper-cleanup): Contribution (4) is “framework”/engineering-heavy; decide whether
// to drop, compress to 1 sentence, or move to appendix / repo documentation.

The remainder of the paper covers related work, problem formulation, dataset
and oracle computation,
// <rm>
// Hedging / inconsistency: we *do* train VINv3; avoid calling it an “implementation sketch”.
an implementation sketch for learning-based scoring,
// </rm>
the evaluation protocol, and diagnostic findings, followed by limitations and
future directions.

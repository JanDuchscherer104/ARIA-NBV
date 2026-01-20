= Introduction

Active 3D reconstruction systems must decide where to move the sensor next in
order to maximize reconstruction quality under limited capture budgets
@VIN-NBV-frahm2025. This next-best-view (NBV) problem is especially challenging
for egocentric platforms where scenes are large, cluttered, and partially
observed. Many classical NBV methods rely on proxy criteria
such as coverage or information gain, which can miss occluded or geometrically
complex regions @VIN-NBV-frahm2025. Recent methods such as VIN-NBV instead optimize reconstruction
quality directly by predicting Relative Reconstruction Improvement (RRI) for
candidate views in object-centric capture settings @VIN-NBV-frahm2025. Our goal
is to bring this quality-driven paradigm to the Aria ecosystem by combining the Aria Synthetic Environments
(ASE) dataset, the EFM3D/EVL foundation model stack, and an oracle RRI
computation pipeline tailored to egocentric trajectories @EFM3D-straub2024
@ProjectAria-ASE-2025. Project Aria provides an egocentric multi-modal recording
platform and toolchain, including calibrated, time-aligned sensor streams and
machine perception services that recover accurate trajectories and semi-dense
reconstructions from raw recordings @projectaria-engel2023.
This ecosystem perspective matters for NBV: it motivates training and debugging
policies on scalable synthetic data while targeting the same geometric
primitives (trajectory + semi-dense reconstruction) expected on real Aria
recordings.

In parallel, reinforcement-learning approaches such as GenNBV learn continuous
5-DoF free-space policies, using multi-source state embeddings and coverage-gain
rewards to improve cross-dataset generalization @GenNBV-chen2024. However,
coverage remains a surrogate for reconstruction quality and can under-prioritize
occlusions and fine details @VIN-NBV-frahm2025. VIN-NBV instead leverages an oracle RRI signal to
train a lightweight candidate-ranking network via imitation learning, making
the objective directly reconstruction-quality aligned @VIN-NBV-frahm2025.

In this work we do not attempt to learn a continuous 5-DoF action policy
end-to-end, nor do we present a learned next-best-view policy yet. Instead, we
focus on the oracle supervision signal itself: evaluating a single step is
expensive because it requires depth rendering and point↔mesh distance
computation across multiple candidate views. Following VIN-NBV, we discretize
the action space via a candidate set and compute per-candidate oracle RRI
labels @VIN-NBV-frahm2025. Learning a view-scoring model on top of these labels
is left to future work.

We target the following setting. Each ASE scene provides a prerecorded Aria
trajectory with RGB and SLAM cameras, semi-dense SLAM points, and a ground-truth
mesh (for a supervised subset). We sample candidate camera poses around the
current rig pose, render candidate depth maps from the mesh, and compute oracle
RRI for each candidate. These labels can supervise a view introspection network
(VIN) that predicts ordinal improvement scores without rendering or acquiring
the candidate view @VIN-NBV-frahm2025.
Compared to coverage-only objectives, this aligns the policy with actual
surface reconstruction quality @VIN-NBV-frahm2025.

We provide a Streamlit-based diagnostics dashboard to inspect candidate
generation, depth rendering, and oracle RRI distributions, which helps validate
coordinate conventions and exposes failure modes (e.g., candidates that look
into empty space). In future work, we plan to train a VIN-style candidate
scorer on top of EFM3D/EVL features, combining local voxel evidence with
explicit view-conditioned cues derived from semi-dense projections and
FiLM-style conditioning @perez2017filmvisualreasoninggeneral @EFM3D-straub2024.

*Contributions.* This paper documents the oracle supervision pipeline and
provides a reproducible baseline for future learning-based NBV policies.

- We define an oracle RRI pipeline for ASE that fuses semi-dense SLAM points
  with candidate depth renderings and evaluates reconstruction quality against
  ground-truth meshes.
- We describe an ordinal (CORAL) label representation and evaluation protocol
  for future learning of RRI predictors @CORAL-cao2019 @VIN-NBV-frahm2025.
- We provide system-level diagnostics and visualization tools that expose
  candidate sampling, depth rendering, and oracle RRI behavior.

The remainder of the paper covers related work, problem formulation, dataset
and oracle computation, an implementation sketch for learning-based scoring,
the evaluation protocol, and diagnostic findings, followed by limitations and
future directions.

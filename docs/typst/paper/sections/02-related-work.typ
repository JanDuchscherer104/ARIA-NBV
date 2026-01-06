= Related Work

== Next-best-view planning

Early NBV systems optimize coverage or entropy-based utility functions. Modern
learning-based methods fall into two categories: continuous-action policies and
discrete candidate ranking. GenNBV learns a continuous 5-DoF policy with
reinforcement learning over occupancy and appearance cues @GenNBV-chen2024.
VIN-NBV instead samples candidate views and predicts RRI via imitation learning
from an oracle, leading to strong gains in reconstruction quality compared to
coverage-only objectives @VIN-NBV-frahm2025. Our approach follows the latter
paradigm, but adapts it to egocentric trajectories and the ASE dataset.

== Egocentric foundation models

EFM3D introduces a benchmark and model stack for egocentric 3D perception,
including the EVL architecture that lifts multi-view observations into a local
voxel grid @EFM3D-straub2024. EVL outputs occupancy, centerness, and OBB
predictions, providing a compact scene representation. We treat EVL as a frozen
backbone and build a lightweight NBV head on top of its voxel features.

#figure(
  image("/figures/efm3d/efm3d_arch_v1.pdf", width: 100%),
  caption: [EFM3D/EVL architecture overview (from the EFM3D release).],
) <fig:efm3d-arch>

== Scene-level representations

SceneScript proposes a structured language for indoor scene layouts and
introduces ASE as a large-scale synthetic dataset with egocentric trajectories
and ground-truth geometry @SceneScript-avetisyan2024. We focus on ASE for oracle
RRI computation and later extensions toward entity-aware NBV. Our work also
relates to Project Aria's broader multimodal dataset and tooling
@projectaria-engel2023 @ProjectAria-ASE-2025.

== Ordinal regression for continuous targets

Predicting RRI benefits from ordinal regression because reconstruction
improvement is a continuous scalar with heavy-tailed distributions. CORAL
models ordinal labels via cumulative probabilities and has been used for age
estimation and related tasks @CORAL-cao2019. We adopt CORAL for RRI binning and
use a calibrated expectation over bins for auxiliary regression.

== Feature-wise conditioning (FiLM)

Feature-wise Linear Modulation (FiLM) applies a learned per-channel scale and
shift to condition intermediate features on an auxiliary signal
@perez2017filmvisualreasoninggeneral. We use FiLM-style conditioning to
modulate voxel-derived candidate features with view-dependent semidense
projection statistics, providing a lightweight mechanism for late fusion.

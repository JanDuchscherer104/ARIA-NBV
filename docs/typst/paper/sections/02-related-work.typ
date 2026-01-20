= Related Work

#import "/typst/shared/macros.typ": *

== Next-best-view planning

Early NBV systems optimize coverage or information-gain style utilities, while
recent learning-based methods largely fall into (i) continuous-action policies
trained with reinforcement learning and (ii) discrete candidate ranking
approaches @VIN-NBV-frahm2025. GenNBV learns a continuous 5-DoF free-space policy
with PPO, using a multi-source state embedding (probabilistic occupancy from
depth, RGB semantics, and action history) and a coverage-gain reward
@GenNBV-chen2024. VIN-NBV instead samples candidate views and predicts Relative
Reconstruction Improvement (RRI) via imitation learning on oracle RRI labels,
yielding improved reconstruction quality compared to coverage-only objectives
@VIN-NBV-frahm2025. Our approach follows the latter paradigm, but adapts it to
egocentric trajectories and the ASE dataset.

== Egocentric foundation models

EFM3D introduces a benchmark for egocentric 3D perception with two core tasks:
3D OBB detection and surface regression, and proposes EVL, which lifts
multi-stream RGB + SLAM snippets into a local, gravity-aligned voxel grid using
frozen 2D foundation features plus semi-dense point and free-space masks
@EFM3D-straub2024. The grid is anchored to the last RGB pose (local ~4 m x 4 m x
4 m extent) and processed by a 3D U-Net @UNet3D-cicek2016 before dense heads
predict occupancy, centerness, box parameters, and class logits;
post-processing yields OBB detections @EFM3D-straub2024. The release also adds
ASE OBB visibility metadata, GT meshes for ASE validation and ADT, and a small
real-world Aria Everyday Objects (AEO) set to support sim-to-real evaluation
@EFM3D-straub2024. We treat EVL as a frozen backbone and build a lightweight NBV
head on top of its voxel features; the local extent motivates our semi-dense
projection cues for out-of-bounds candidates.

#figure(
  image("/figures/efm3d/efm3d_arch_v1.pdf", width: 100%),
  caption: [EFM3D/EVL architecture overview (from the EFM3D release) @EFM3D-straub2024.],
) <fig:efm3d-arch>

== Scene-level representations

SceneScript proposes a structured language for indoor scene layouts and
introduces ASE as a large-scale synthetic dataset with egocentric trajectories
and ground-truth geometry @SceneScript-avetisyan2024. We focus on ASE for oracle
RRI computation and later extensions toward entity-aware NBV. Our work also
relates to Project Aria's broader multimodal dataset and tooling
@projectaria-engel2023 @ProjectAria-ASE-2025. Project Aria introduced an
egocentric multi-modal recording device with a rich sensor suite and a
standardized software stack: recordings are stored as synchronized streams
(VRS container) with factory calibration, and machine perception services
recover accurate trajectories and semi-dense point clouds aligned in a global
frame @projectaria-engel2023. An open C++/Python toolkit provides access to VRS
recordings, calibration, and MPS outputs for downstream research
@projectaria-engel2023. The underlying hardware is designed around a
multi-camera + IMU stack for egocentric perception, and MPS additionally
provides online calibration to account for small deformations while wearing the
device and time-alignment mechanisms for multi-device capture
@projectaria-engel2023. These components define the Aria ecosystem that ASE
and EFM3D build upon and that our NBV pipeline targets.

== Ordinal regression for continuous targets

Oracle RRI is a continuous target, but NBV ultimately requires a robust
#textit[ranking] of candidate views. VIN-NBV reports that directly regressing RRI
is challenging and can hurt generalization; large outliers and stage-dependent
scaling make absolute-value regression brittle when only the relative ordering
matters @VIN-NBV-frahm2025. A common remedy is to discretize RRI into $K$
ordered bins and pose prediction as #textit[ordinal] classification: the label
$y in {0, ..., K-1}$ has a natural order, and misclassifying a candidate by
many bins should be penalized more than confusing nearby bins.
Unlike nominal $K$-way classification, this setting can exploit label ordering
instead of treating bins as unrelated categories @CORAL-cao2019.

Among ordinal losses, CORAL is attractive because it is #textit[rank-consistent]
and efficient @CORAL-cao2019. It converts the $K$-class ordinal problem into
$K-1$ binary threshold tasks (predicting whether $y$ exceeds each rank) with
shared classifier weights, which avoids contradictory non-monotone outputs that
can arise from independent one-vs-rest reductions. CORAL therefore yields
well-structured cumulative probabilities that can be mapped back to a scalar
score (e.g., via an expected bin value) for candidate ranking while reducing
large misclassifications.

== Feature-wise conditioning (FiLM)

Feature-wise Linear Modulation (FiLM) applies a learned per-channel scale and
shift to condition intermediate features on an auxiliary signal
@perez2017filmvisualreasoninggeneral. We use FiLM-style conditioning to
modulate voxel-derived candidate features with view-dependent semi-dense
projection statistics, providing a lightweight mechanism for late fusion.

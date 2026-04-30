= Related Work

#import "../../shared/macros.typ": *

== Next-best-view planning

// TODO(paper-cleanup): The “early NBV systems optimize coverage/info gain” claim is broad;
// either add representative citations beyond VIN-NBV or narrow phrasing.
// TODO(paper-cleanup): Avoid re-explaining GenNBV/VIN-NBV here if Intro already does; keep
// to 1–2 crisp sentences + citations and defer details to later sections.

Early NBV systems optimize coverage or information-gain utilities, while
learning-based methods largely fall into (i) continuous-action policies trained
with reinforcement learning and (ii) discrete candidate-ranking approaches
@VIN-NBV-frahm2025. GenNBV learns a continuous 5-DoF free-space policy with PPO
and coverage-gain rewards @GenNBV-chen2024. VIN-NBV instead samples candidate
views and predicts Relative Reconstruction Improvement (RRI) via imitation
learning on oracle labels @VIN-NBV-frahm2025.

== Egocentric foundation models

// TODO(paper-cleanup): Verify EFM3D task framing (“two core tasks”) matches the paper; avoid
// overspecific claims unless directly cited.
// TODO(paper-cleanup): The last two sentences are *our* method choice (EVL frozen + semidense
// cues); move to Architecture to keep Related Work descriptive, not prescriptive.

EFM3D introduces a benchmark for egocentric 3D perception with two core tasks:
3D OBB detection and surface regression, and proposes EVL, which lifts
multi-stream RGB + SLAM snippets into a local, gravity-aligned voxel grid using
frozen 2D foundation features plus semi-dense point and free-space masks
@EFM3D-straub2024. The grid is anchored to the last RGB pose (local 4 m cube; in
the voxel frame the extent is $[-2, 2] #sym.times [0, 4] #sym.times [-2, 2]$) and
processed by a 3D U-Net @UNet3D-cicek2016 before dense heads predict occupancy,
centerness, box parameters, and class logits;
post-processing yields OBB detections @EFM3D-straub2024. The release also adds
ASE OBB annotations and GT meshes for the validation subset, as well as a small
real-world Aria Everyday Objects (AEO) set to support sim-to-real evaluation
@EFM3D-straub2024. We treat EVL as a frozen backbone and build a lightweight NBV
head on top of its voxel features; the local extent motivates semi-dense
projection cues for out-of-bounds candidates.
// <rm>
// The parenthetical “local 4m cube” reads like a universal EVL property; prefer phrasing as
// “in our configuration …” and moving exact extent/resolution to a baseline/config table.
// The last two sentences (“we treat EVL as frozen …”) are method choices and belong in the
// Architecture section, not Related Work.
// </rm>

#figure(
  image("/figures/external/arXiv-EFM3D/efm3d_arch_v1.pdf", width: 100%),
  caption: [EFM3D/EVL architecture overview (from the EFM3D release) @EFM3D-straub2024.],
) <fig:efm3d-arch>

== Ordinal regression for continuous targets

// TODO(paper-cleanup): This duplicates Training Objective; consider keeping only the high-level
// motivation here and refer to Section 7 for equations/implementation.
// TODO(paper-cleanup): Align notation with macros: use `RRI` macro consistently and use bold(...)
// for vectors/probabilities (or reuse #eqs where possible).

// <rm>
// Duplicates Section 7 (Training Objective). Keep only a 1–2 sentence pointer here or remove.
Oracle RRI is a continuous target, but NBV ultimately requires a robust
#textit[ranking] of candidate views. Directly regressing RRI
is challenging and can hurt generalization; heavily skewed RRI distributions with large outliers and stage-dependent scaling make absolute-value regression brittle @VIN-NBV-frahm2025. A common remedy is to discretize RRI into $K$
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
// </rm>

== Feature-wise conditioning (FiLM)

// TODO(paper-cleanup): This is currently written as “we use FiLM …” (method detail);
// move to Architecture or rephrase as related-work background only.
// TODO(paper-cleanup): “varies heavily across candidates” needs a concrete diagnostic/figure
// reference or should be softened.

Feature-wise Linear Modulation (FiLM) applies a learned per-channel scale and
shift to condition intermediate features on an auxiliary signal
@perez2017filmvisualreasoninggeneral. We use FiLM-style conditioning to
modulate voxel-derived candidate features with view-dependent semi-dense
projection statistics, providing a lightweight mechanism for late fusion when reliability of the modulated signal varies heavily across candidates.
// <rm>
// Method-specific sentence (“We use …”) belongs in Architecture; keep this subsection as
// related work background only.
// </rm>

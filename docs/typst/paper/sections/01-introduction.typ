= Introduction

Active 3D reconstruction systems must decide where to move the sensor next in
order to maximize reconstruction quality. This next-best-view (NBV) problem is
especially challenging for egocentric platforms where scenes are large,
cluttered, and partially observed. Recent methods such as VIN-NBV optimize
reconstruction quality directly by predicting Relative Reconstruction
Improvement (RRI) for candidate views @VIN-NBV-frahm2025. Our goal is to bring
this quality-driven paradigm to the Aria ecosystem by combining the Aria
Synthetic Environments (ASE) dataset, the EFM3D/EVL foundation model stack, and
an oracle RRI computation pipeline tailored to egocentric trajectories
@EFM3D-straub2024 @ProjectAria-ASE-2025.

In parallel, reinforcement-learning approaches such as GenNBV demonstrate how
generalizable NBV policies can operate in large free-space action domains, but
they typically optimize surrogate rewards and require careful state
representations to scale beyond constrained object-centric capture
@GenNBV-chen2024. VIN-NBV instead leverages an oracle RRI signal to train a
lightweight candidate ranking network, making the objective directly
reconstruction-quality aligned @VIN-NBV-frahm2025.

We target the following setting. Each ASE scene provides a prerecorded Aria
trajectory with RGB and SLAM cameras, semi-dense SLAM points, and a ground-truth
mesh. We sample candidate camera poses around the current rig pose, render
candidate depth maps from the mesh, and compute the oracle RRI for each
candidate. These RRI labels train a view introspection network (VIN) that
predicts ordinal improvement scores without rendering the candidate view.
Compared to coverage-only objectives, this aligns the policy with actual
surface reconstruction quality.

Our current Aria-VIN-NBV implementation integrates a frozen EVL backbone and a
lightweight prediction head that conditions on candidate pose, EVL voxel
features, and semidense view conditioning. The model incorporates
pose-conditioned global pooling, candidate-conditioned semidense projection
statistics, and frustum-aware cross-attention over projected semidense points.
We further include a trajectory encoder for history context and a voxel
reliability proxy (a voxel-validity fraction) that gates global features when
candidates drift outside EVL's voxel extent. Optional PointNeXt-based semidense
embeddings provide an additional global geometry cue, and we use FiLM-style
feature modulation to condition voxel-derived features on view-dependent
semidense statistics @perez2017filmvisualreasoninggeneral.

*Contributions.* This paper documents the state of the system and provides a
reproducible baseline for future ablations.

- We define an oracle RRI pipeline for ASE that fuses semi-dense SLAM points
  with candidate depth renderings and evaluates reconstruction quality against
  ground-truth meshes.
- We introduce a VIN v2 architecture that combines EVL voxel features, pose
  encodings, semidense projection statistics, frustum-aware attention, and
  optional trajectory/semidense encoders for candidate-dependent scoring.
- We describe the training objective based on CORAL ordinal regression and
  auxiliary regression for calibrated RRI estimates @CORAL-cao2019.
- We provide system-level diagnostics and visualization tools that expose
  candidate sampling, depth rendering, and per-module gradients.

The remainder of the paper covers related work, problem formulation, dataset
and oracle computation, the VIN architecture, training objective, and current
diagnostic findings, followed by limitations and future directions.

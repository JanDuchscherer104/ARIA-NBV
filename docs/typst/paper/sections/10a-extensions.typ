#import "../../shared/macros.typ": *



= Future Extensions <sec:entity-aware>

This project currently establishes oracle supervision for egocentric NBV and a
learned candidate scorer baseline. The following extensions are enabled by the
existing pipeline and represent the main research and engineering directions we have identified.

== Toward Entity-Aware NBV <sec:toward-entity-aware>

ASE provides object-level annotations and EVL predicts 3D OBBs, enabling
task-driven NBV. Instead of optimizing only scene-level reconstruction quality,
we can optimize a weighted combination of global and entity-specific
improvement:

#block[#align(center)[#eqs.entity.objective]]

Here, $(#(symb.entity.w) _e)$ encodes entity importance (user- or task-defined),
and #symb.entity.lambda_scene trades off between entity-centric and global
quality.

*How to define the entity-specific term.* A practical definition reuses the
existing oracle and changes only the *evaluation subset*:

- *OBB-cropped mesh + points:* define an entity region from a GT OBB (or EVL
  prediction), crop #symb.ase.mesh and filter both the current semidense points
  $#(symb.ase.points_semi) _t$ and candidate points #symb.oracle.points_q to that
  region (plus a margin), then compute $#(symb.oracle.rri) _e(q)$ with the same
  point #sym.arrow.l.r mesh evaluation.
- *OBB surface proxy:* approximate the entity surface by sampling points on the
  OBB (or an OBB-derived SDF shell) and compute a completeness-style proxy that
  rewards observing previously missing regions of that proxy surface.

== Scaling supervision and data products

Oracle labels are expensive; scaling coverage is the primary lever for stronger
training and more reliable conclusions.

- Address dataset constraints explicitly: each ASE scene provides only one
  prerecorded trajectory (no arbitrary novel viewpoints), GT meshes are limited
  to a subset of scenes, and beyond that we must rely on pseudo-GT or reduced
- Extend the offline cache to cover the full mesh-supervised ASE subset.
- While the number of scenes with GT meshes is limited to 4608 (of which we have been using 19.2%), we can increase variability by:
  - Choosing subsequences of the pre-recorded trajectories
  - Generating multiple candidate sets per snippet and altering the candidate
    sampling strategy (e.g., more diverse orientations, wider spatial coverage, allow backward-facing views, roll jitter, ...)

== From discrete candidates to continuous planning

The current system ranks a discrete candidate set; planning and continuous
action spaces remain open.

- Learn continuous pose proposal distributions whose samples are scored by the
  learned RRI predictor, using free-space/occupancy for collision avoidance.
- Evaluate the design choice of *filtering* invalid candidates versus including
  them with strong penalties; this affects the RRI distribution and therefore
  CORAL binning and calibration. There are various caveats to consider.
- Explore differentiable refinement of candidate poses by querying voxel fields
  with differentiable sampling primitives, enabling gradient-based local pose
  improvement around promising candidates.

== Oracle throughput and robustness

Several improvements target correctness and scalability of oracle supervision.

== Upgrades to the View Introspection Network

The main goal is to strengthen candidate-specific signal, improve calibration,
and reduce mode collapse while preserving interpretability.

- Apply candidate shuffling in the datamodule after batching.
- Validate learnable CORAL bin shifts/centers against baseline fixed binning.

== Evaluation, deployment, and human-in-the-loop systems

Beyond per-snippet ranking, we need end-to-end evaluation and real-world
integration.

- Extend toward real-device deployment (Aria, Quest 3, iPhone LiDAR) and
  sim-to-real evaluation using the same pose/camera/point primitives.
- Compare alternative backbones for NBV scoring (e.g., EVL vs. SceneScript) and
  assess whether fine-tuning an EFM on NBV objectives improves performance on
  target platforms.
- Build an interactive human-in-the-loop NBV guidance system: entity selection
  UI, real-time scoring with streaming updates, and AR overlays for viewpoint
  guidance.
- Explore LLM/VLA integration for natural-language explanations and high-level
  task guidance layered on top of entity-aware objectives to allow specifying objects of interest via language.

== Experiment management and reporting

- Run stationary Optuna sweeps focused on architectural toggles (avoid
  confounding schedule/width changes) and tag trials by sweep phase for clean
  analysis.

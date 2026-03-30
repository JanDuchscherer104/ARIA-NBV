---
id: project_state
updated: 2026-03-29
scope: repo
owner: jan
status: active
tags: [nbv, rri, efm3d, ase, thesis]
---

# Project State

## Mission
This repository develops a quality-driven next-best-view system for egocentric indoor scenes in the Aria ecosystem. Instead of optimizing proxy objectives such as coverage, the project uses Relative Reconstruction Improvement (RRI) to rank candidate views by their expected effect on reconstruction quality. The strategic motive is to use expensive oracle supervision in simulation to train a lightweight scorer that can later support stronger planning, broader supervision, and sim-to-real transfer.

## Current System
- Offline oracle label pipeline on ASE/EFM snippets: sample discrete candidate poses around the latest rig pose, prune invalid views, render metric depth from GT meshes, backproject candidate point clouds, fuse them with semi-dense SLAM points, and score candidates with Chamfer-derived RRI plus accuracy/completeness terms.
- Reproducible offline cache of oracle-labelled snippets, candidate metadata, optional candidate renders and point clouds, and frozen EVL/EFM features for batched training.
- Implemented VIN v3 baseline: frozen EVL voxel backbone, rig-relative pose encoding, pose-conditioned global context, candidate-specific semidense projection evidence, optional projection-grid features, and a CORAL ordinal head for RRI prediction.
- Research tooling around the pipeline: Lightning experiment configs, cache builders, CLIs, W&B/Optuna hooks, and Streamlit diagnostics for candidate generation, rendering, projection, and oracle-label behavior.

## Current Research Position
- The project currently solves one-step discrete candidate ranking, not end-to-end NBV control.
- The learned model predicts offline oracle supervision; it is not yet a continuous action policy or a multi-step planner.
- EVL is currently used as a frozen backbone. The main learning effort is in the candidate scorer, feature design, and label representation.
- The present baseline is useful and trained, but still exploratory. It supports diagnostics and ablations more than final deployment claims.

## Implemented Ground Truth
- Oracle RRI computation is the canonical supervision signal.
- Candidate rendering and backprojection are implemented with PyTorch3D plus Aria/EFM geometric primitives.
- Semi-dense SLAM points, GT meshes, candidate cameras, and typed Aria poses are the key geometric inputs for oracle scoring.
- Offline cache generation and inspection are first-class parts of the workflow because oracle labels are too expensive to recompute inside normal training loops.

## Active Work
- Improve the VIN scorer: stronger candidate-specific features, better calibration, candidate shuffling, CORAL/binning refinements, and updated architectures.
- Scale oracle supervision: cover more mesh-supervised ASE data, generate broader candidate distributions, and clean up data handling.
- Keep coordinate and frame handling correct across sampling, rendering, caching, and visualization, especially around gravity alignment and `rotate_yaw_cw90`.
- Keep Quarto docs aligned with the current codebase, `docs/typst/paper/main.typ`, and the thesis slides.

## Current Constraints and Risks
- GT-mesh supervision covers only a limited mesh-supervised subset of ASE, not the full dataset.
- Oracle labels remain expensive because each snippet requires many candidate renders and point-to-mesh evaluations.
- EVL's voxel extent is local around the latest pose, so far-away candidates need extra candidate-specific evidence.
- Candidate-frame consistency is easy to break; display-time Aria rotations must not leak into physical geometry or cached training inputs.
- Overfitting, calibration drift, and stage-dependent RRI distributions remain open concerns for the current VIN setup.

## Thesis-Driven Next Directions
- Decide how much to invest in an updated VIN versus moving sooner toward multi-step RL.
- Explore multi-step, non-myopic planning where the learned RRI scorer can serve as a surrogate objective or critic.
- Investigate richer counterfactual modalities for non-trajectory poses, including simulator-generated signals or one-shot reconstruction methods.
- Extend from scene-level reconstruction quality toward entity-aware or task-aware NBV objectives.

## Pointers
- Paper ground truth: `docs/typst/paper/main.typ`
- Immediate implementation tasks and unresolved issues: `docs/contents/todos.qmd`
- Open research questions: `.agents/memory/state/OPEN_QUESTIONS.md`
- Longer-horizon scratchpad and thesis ideas: `ideas.md`

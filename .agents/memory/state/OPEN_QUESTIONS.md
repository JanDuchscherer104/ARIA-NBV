---
id: open_questions
updated: 2026-03-31
scope: repo
owner: jan
status: active
tags: [research, nbv, vin, training]
---

# Open Questions

## Advisor-Facing Scope and Priority Questions
- Should the thesis be scoped around geometry-first non-myopic planning on the current ASE / EFM stack, or should a major share of time shift toward a VIN v4 or broader model-architecture rewrite?
- Should the thesis core stay within the current mesh-supervised ASE ecosystem, or is there a strong reason to switch datasets, simulators, or supervision sources earlier?
- Which external asks are worth escalating now: university workstation access, ASE simulator access, Aria Gen2 hardware, or some narrower subset?
- After bounded oracle-RRI lookahead is compared with one-step greedy, should the next non-myopic baseline be beam search, close-greedy control, or discrete-shell RL?

## Planning and RL Formulation
- Which counterfactual modalities are acceptable in the first multi-step setting: GT mesh, SDF / visibility, normals, depth, synthetic SLAM, splats, or learned world models?
- Can the critic use privileged GT signals such as OBBs, segmentation masks, or mesh-derived descriptors during training if the actor cannot?
- Should the first RL variant be explicitly close-greedy, low-discount, and receding-horizon rather than full long-horizon RL?
- When should the current discrete shell give way to hierarchical target-selection plus view realization or to continuous pose control?
- How should invalid or infeasible actions be handled: hard masking, projected feasibility, penalties, or a mixed strategy?
- If semantic-global planning is pursued later, what grounded world-memory schema and verifier / replanner loop would be required?

## Data, Supervision, and Robustness
- Should fine-detail oracle experiments ban aggressive mesh or point-cloud downsampling outright?
- How much of the 4608 mesh-supervised snippets should be brought into the immutable VIN offline store before model complexity increases further?
- Is broader candidate generation more valuable right now than deeper VIN changes: more than 60 candidates, more anchor poses, wider azimuth coverage, or roll / backward-view variants?
- How should stage dependence and label-distribution drift be handled: stage-aware features, dynamic binning, calibration analysis, or some combination?
- What exactly caused the apparent overfitting or calibration-failure signals, including collapse in the lowest ordinal classes?

## VIN and Representation Questions
- Which candidate-specific signals should be prioritized next: directional observability, target-conditioned local reads, stronger projection encoders, or transformer-style query-centric fusion?
- Which current VIN components actually help enough to keep: surface reconstruction input, modified CORAL, auxiliary Huber loss, pretrained projection encoders?
- Should entity-aware or object-centric supervision become part of the thesis core or remain a phase-2 extension?

## System Questions
- Where should gravity-aligned sampling convenience end and physical rig-frame supervision begin?
- Which diagnostics and acceptance checks are strong enough to catch pose-frame mismatches early?
- What is the minimum stable context bundle Codex needs for code, paper, and experiment tasks without reintroducing large static dumps?

## Recently Locked Decisions
- The first non-myopic comparison is bounded oracle-RRI lookahead versus one-step greedy under equal budget.
- The first rollout actor-visible state is geometry-first; oracle labels, GT mesh quantities, and GT OBB crops are supervision/evaluation only.
- Multi-step reward starts with episode-normalized additive RRI or log-improvement reward, while the current one-step RRI label remains compatible with VIN training.
- The first target-aware metric is GT-OBB-cropped target RRI.
- The VIN evidence gate is held-out ranking, oracle-evaluated VIN-selected rollouts, calibration/stage-shift analysis, Rerun failure visualization, and only then value, Q, or return heads.

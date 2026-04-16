---
id: project_state
updated: 2026-04-15
scope: repo
owner: jan
status: active
tags: [nbv, rri, efm3d, ase, codex]
---

# Project State

## Goal and Core Claim
This repository develops an active next-best-view planner for egocentric indoor scenes. The central claim is that ranking viewpoints by relative reconstruction improvement (RRI) is a stronger objective than proxy goals such as pure coverage, especially when candidate scoring is grounded in ASE meshes, semi-dense reconstruction state, and frozen egocentric foundation-model features.

## Current Thesis Spine
- *Current truth / active direction:* The implemented stack is still centered on the current ASE + EFM ecosystem: oracle RRI label generation for discrete candidate sets, VIN-style learned scoring on frozen backbone features, and diagnostics around candidate generation, rendering, and surface-error behavior.
- *Current truth / active direction:* Non-myopic work is now present but still incremental. The repo supports multi-step counterfactual rollouts with structured evaluator metrics, cumulative-RRI accounting, cumulative-RRI plotting, and a first Gymnasium / Stable-Baselines3 RL scaffold over a fixed candidate shell.
- *Current truth / active direction:* The learned VIN remains a preliminary candidate scorer rather than a full next-best-view policy. The strongest current story is therefore geometry-first NBV research with growing planning capability, not a finished end-to-end RL system.
- *Current truth / active direction:* Offline caches, W&B + Optuna integration, Streamlit inspection tools, and the Typst / Quarto reporting stack are in place to support controlled ablations and advisor-facing iteration.

## Ranked Priorities and Meeting Decisions
1. *Thesis core and scope lock.* *Status:* Open question / supervisor decision. *Current state:* the clearest implemented narrative is RRI-driven non-myopic planning on the current ASE / EFM stack; broad policy learning, semantic planning, and deployment remain extensions. *Why it matters now:* this determines whether the thesis stays coherent or fragments across too many fronts. *Recommended direction:* keep the thesis centered on quality-driven non-myopic planning and treat larger RL / systems ambitions as staged follow-up work.
2. *First non-myopic milestone.* *Status:* Open question / supervisor decision. *Current state:* the codebase already has beam-style counterfactual rollout machinery and a discrete-shell RL scaffold, while continuous 5-DoF policy learning still lacks a tractable reward/evaluation loop. *Why it matters now:* this sets the next experiment family. *Recommended direction:* prioritize search, close-greedy control, or discrete-shell RL before any continuous controller.
3. *Stay within the current ecosystem and dataset.* *Status:* Open question / supervisor decision. *Current state:* the oracle, VIN, and rollout stack all rely on the present ASE / mesh-supervised setup, and the paper still frames this ecosystem as the project's main evidence base. *Why it matters now:* switching ecosystems would delay thesis progress and weaken comparability. *Recommended direction:* stay on the current mesh-supervised ASE subset for the thesis core, then expand only after the non-myopic baseline is clear.
4. *How much time VIN v4 should get versus planning.* *Status:* Open question / supervisor decision. *Current state:* VIN v3 is preliminary and there are many plausible upgrades, but the repo already shows that planning and reward definition are the larger unresolved story. *Why it matters now:* unchecked VIN work can turn the thesis into an architecture search project. *Recommended direction:* keep VIN work bounded to controlled ablations and the most plausible view-conditioned improvements.
5. *Infrastructure and access priorities.* *Status:* Open question / supervisor decision. *Current state:* oracle throughput, local compute friction, and limited simulator access slow iteration more than model plumbing does. *Why it matters now:* external asks should be focused and defensible. *Recommended direction:* prioritize university workstation access first, ASE simulator access second, and hardware requests such as Aria Gen2 after that.
6. *Evidence bar for RL / planning claims.* *Status:* Open question / supervisor decision. *Current state:* the paper still positions full policy learning as future work, and current rollout / RL results are scaffolding rather than thesis-grade evidence. *Why it matters now:* it defines what has to be measured before making stronger claims. *Recommended direction:* require oracle-throughput measurements, calibration / stage-shift analysis, and one-step versus non-myopic comparisons before claiming RL progress.
7. *MDP contract and modality split.* *Status:* Open question / supervisor decision. *Current state:* full modalities are only available on logged ego-trajectories, while counterfactual states currently rely on geometry-derived quantities and accumulated selected observations. *Why it matters now:* this is the central design decision for multi-step learning. *Recommended direction:* keep the first actor geometry-first, allow richer or privileged signals only where explicitly justified, and make the historical-versus-counterfactual split explicit.
8. *Supervision scaling within existing GT coverage.* *Status:* Open question / supervisor decision. *Current state:* only part of the mesh-supervised ASE subset has been used for training, and the current candidate budget is still narrow. *Why it matters now:* scaling supervision may buy more than architectural novelty. *Recommended direction:* expand within the existing GT subset first through more snippets, more anchor poses, more candidate sets, and broader candidate distributions.
9. *Hierarchical RL as a phase-2 branch.* *Status:* Open question / supervisor decision. *Current state:* Hestia-style ideas map well onto the project, but the current repo only supports the first geometry-first scaffolding. *Why it matters now:* hierarchy is promising, but premature commitment could distract from getting the baseline right. *Recommended direction:* treat target-selection-plus-view-realization as an important phase-2 extension after discrete non-myopic baselines are stable.
10. *Entity-aware RRI and farther-horizon extensions.* *Status:* Open question / supervisor decision. *Current state:* ASE already exposes object-level annotations and OBBs, and the paper identifies entity-aware objectives and semantic-global planning as strong follow-up directions. *Why it matters now:* these are attractive ways to widen impact without changing the core thesis claim. *Recommended direction:* keep them explicitly in scope as extensions, but not as blockers for the main geometry-first story.

## Current Issues and Blockers
- *Issue / blocker:* Fine-detail supervision is fragile if meshes or point clouds are downsampled aggressively. Older runs downsampled the mesh to 10% of faces, which can erase the very surface detail that RRI should reward.
- *Issue / blocker:* Calibration and label-distribution issues remain unresolved. Stage dependence, possible overfitting, and collapse in the lowest ordinal classes are all plausible explanations, and they are not yet cleanly separated.
- *Issue / blocker:* Candidate-generation realism versus generalization is still under-specified. Tight azimuth / elevation bounds improve realism, but they may also bias the learned policy and under-expose exploratory views.
- *Issue / blocker:* Oracle throughput is still the main scaling bottleneck. The paper explicitly notes that oracle cost makes direct on-policy continuous control impractical in the current system.
- *Issue / blocker:* Counterfactual multi-step states still lack full RGB, SLAM, and semantic modalities unless they are synthesized or approximated from geometry.
- *Issue / blocker:* Engineering friction still matters: data-handling cleanup, doc synchronization, and better compute access directly affect iteration speed even though they are not thesis contributions by themselves.

## Near-Term Next Steps
- *Current truth / active direction:* The agent scaffold now uses a thin-root, local-delta guide structure with repo-specific context split across narrower skills and an agent/tooling DB for scaffold debt.
- *Current truth / active direction:* Use discrete search, beam rollouts, and close-greedy / low-discount RL as the first non-myopic milestone before attempting full continuous control.
- *Current truth / active direction:* Make the multi-step return proxy explicit around cumulative oracle RRI first, then test whether VIN can become a fast surrogate reward or critic.
- *Current truth / active direction:* Scale within the current ecosystem before switching worlds: cover more of the mesh-supervised subset, use more anchor poses, and generate broader candidate sets per snippet.
- *Current truth / active direction:* Keep VIN improvements bounded and evidence-driven: ablate surface reconstruction, CORAL modifications, and auxiliary losses before committing to larger architectural rewrites.
- *Current truth / active direction:* Preserve the current geometry-first interpretation of counterfactual state while clarifying which missing modalities should be synthesized later.
- *Current truth / active direction:* Keep docs, Streamlit surfaces, and canonical memory aligned with code so advisor meetings stay grounded in the actual repo state.

## Deferred but Important Extensions
- *Deferred extension:* Hestia-style hierarchical control that separates target selection from view realization and later introduces continuous motion.
- *Deferred extension:* Entity-aware or task-aware RRI that mixes scene-level and object-level improvement.
- *Deferred extension:* Counterfactual modality synthesis through Gaussian splats, synthetic SLAM, or world-model-style reconstruction for richer multi-step state.
- *Deferred extension:* Semantic-global planning with grounded subgoals such as portals, frontiers, entities, and regions layered on top of the geometric controller.
- *Deferred extension:* Real-device deployment, sim-to-real evaluation, and human-in-the-loop viewpoint guidance systems.

## Pointers
- Stable and already-adopted project choices live in `DECISIONS.md`.
- Advisor-pending scope calls and unresolved research/system questions live in `OPEN_QUESTIONS.md`.
- The highest-signal idea scratchpad remains `docs/contents/ideas.qmd`.

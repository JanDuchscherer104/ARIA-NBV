---
id: project_state
updated: 2026-05-06
scope: repo
owner: jan
status: active
tags: [nbv, rri, efm3d, ase, codex]
---

# Project State

## Goal and Core Claim
This repository develops **ARIA-NBV**, an active next-best-view planner for egocentric indoor scenes. The central claim is that target-conditioned, quality-driven viewpoint selection by relative reconstruction improvement (RRI) is a stronger objective than proxy goals such as pure coverage, especially when candidate scoring and rollout supervision are grounded in ASE meshes, semi-dense reconstruction state, and frozen egocentric foundation-model features.

## Current Thesis Spine
- *Current truth / active direction:* The implemented substrate is still centered on the current ASE + EFM ecosystem: oracle RRI label generation for discrete candidate sets, VIN-style learned one-step scoring on frozen backbone features, rollout scaffolding, and diagnostics around candidate generation, rendering, Rerun inspection, and surface-error behavior.
- *Current truth / active direction:* The thesis plan now has a hard gated core beyond one-step scoring: V0/V1 target contracts, observed-only target selection, mixed observed candidate sets, stable ASE oracle rollouts, and a target-conditioned candidate-query Transformer `Q_H` over finite candidate sets.
- *Current truth / active direction:* The learned VIN remains a preliminary one-step candidate scorer rather than a full next-best-view policy. The required M5 policy-like result is `Q_H`, trained from ASE oracle rollout data to predict one masked bounded-horizon Q value per candidate; full continuous control, online RL, and real-device deployment remain bridge/future work.
- *Current truth / active direction:* Offline caches, W&B + Optuna integration, Streamlit inspection tools, and the Typst / Quarto reporting stack are in place to support controlled ablations and advisor-facing iteration.

## Ranked Priorities and Meeting Decisions
1. *Thesis core and scope lock.* *Status:* Locked locally for the proposal pass. *Current state:* the thesis core is target-conditioned, quality-driven NBV on ASE/EFM with strict M1 contracts, observed-target V1 supervision, ASE oracle rollouts, and a candidate-query Transformer `Q_H`. *Why it matters now:* this prevents the thesis from fragmenting across model search, external simulation, and continuous control. *Decision:* keep Habitat/Isaac, SceneScript, real-device guidance, and continuous actor-critic work as stretch or M6 bridge design.
2. *First non-myopic milestone.* *Status:* Locked for the first comparison. *Current state:* the codebase already has beam-style counterfactual rollout machinery, while continuous 5-DoF policy learning still lacks a tractable online reward/evaluation loop. *Why it matters now:* this sets the next experiment family. *Decision:* compare bounded oracle-RRI lookahead against one-step greedy under equal acquisition and candidate budget, then use oracle lookahead as the upper bound for Q_H.
3. *Hard Q_H deliverable.* *Status:* Locked as M5 thesis core. *Current state:* `Q_H` is not implemented yet; rollout data and storage contracts are prerequisites. *Why it matters now:* the thesis success bar is no longer only a deterministic rollout comparison. *Decision:* train a target-conditioned candidate-query Transformer over finite candidate sets from ASE oracle rollout data and require it to beat one-step greedy/model scoring on cumulative target RRI under equal acquisition budget after oracle evaluation of the selected actions.
4. *Stay within the current ecosystem and dataset.* *Status:* Locked for thesis core. *Current state:* the oracle, VIN, and rollout stack all rely on the present ASE / mesh-supervised setup, and the paper still frames this ecosystem as the project's main evidence base. *Why it matters now:* switching ecosystems would delay thesis progress and weaken comparability. *Decision:* use the ASE mesh/oracle counterfactual rollout loop as the simulator substrate; external simulators are expansion gates.
5. *Target contract and actor visibility.* *Status:* Locked for M3/M4/M5. *Current state:* GT OBB crops are the trustworthy label/evaluation mechanism, but they cannot be the actor-visible input for the main result. *Why it matters now:* using GT as policy input would overstate deployability. *Decision:* V0 uses GT OBB input for sanity/upper-bound runs; V1 OBS-SEL / PRED-Q / GT-EVAL uses observed/predicted OBB inputs matched to GT target-RRI labels.
6. *Infrastructure and access priorities.* *Status:* Locked as M2/M3 gate. *Current state:* oracle throughput, storage pressure, and full-scale generation are too large for ad hoc local loops. *Why it matters now:* full 100-scene / 4,608-snippet coverage is mandatory for final experiments after small-subset correctness. *Decision:* LRZ deterministic sharding, Slurm/DSS staging, Zarr-first rollout/Q storage, and resume-safe writes are hard gates before full-scale generation; do not assume a workstation path.
7. *Invalidity and masks.* *Status:* Locked for M1/M3/M5. *Current state:* invalid candidate cases can otherwise poison ordinal labels and Q targets. *Why it matters now:* invalidity is a constraint, not low reconstruction quality. *Decision:* use hard masks and explicit reason codes; validity heads and scalar penalties are ablations after masks/reasons are stable.
8. *Supervision scaling within existing GT coverage.* *Status:* Locked with coverage reporting. *Current state:* only part of the mesh-supervised ASE subset has been used for training. *Why it matters now:* scaling supervision may buy more than architectural novelty. *Decision:* use small trusted subsets for correctness, then scale to all 100 GT-mesh scenes / 4,608 snippets or use a scene-level held-out subset only with explicit reporting of scenes, snippets, targets, trajectories, rollout seeds, transitions, blockers, and coverage gaps.
9. *VIN and representation work.* *Status:* Bounded support work. *Current state:* VIN v3 is preliminary and there are many plausible upgrades, but unchecked VIN work can become architecture search. *Why it matters now:* Q_H needs a one-step model comparator, not an unlimited model rewrite. *Decision:* keep VIN work to controlled calibration/ablation and target-conditioned scoring needed for M4/M5.
10. *Hierarchical/semantic extensions.* *Status:* Stretch / bridge. *Current state:* Hestia-style hierarchy and SceneScript-style semantics remain promising but not required. *Why it matters now:* they can distract from the finite-candidate Q_H thesis result. *Decision:* document actor-critic/continuous and semantic-global bridges only after M5 evidence is stable.

## Current Issues and Blockers
- *Issue / blocker:* Fine-detail supervision is fragile if meshes or point clouds are downsampled aggressively. Older runs downsampled the mesh to 10% of faces, which can erase the very surface detail that RRI should reward.
- *Issue / blocker:* Calibration and label-distribution issues remain unresolved. Stage dependence, possible overfitting, and collapse in the lowest ordinal classes are all plausible explanations, and they are not yet cleanly separated.
- *Issue / blocker:* Candidate-generation realism versus generalization is still under-specified. Tight azimuth / elevation bounds improve realism, but they may also bias the learned policy and under-expose exploratory views.
- *Issue / blocker:* Oracle throughput is still the main scaling bottleneck. The paper explicitly notes that oracle cost makes direct on-policy continuous control impractical in the current system.
- *Issue / blocker:* `.configs/offline_only.toml` now points at a local `vin_offline` store with manifest, sample index, split arrays, and shards, but the store is partial/interrupted. Local smoke is blocked on corrected command behavior and validation rather than a known missing manifest.
- *Issue / blocker:* Counterfactual multi-step states still lack full RGB, SLAM, and semantic modalities unless they are synthesized or approximated from geometry.
- *Issue / blocker:* Engineering friction still matters: data-handling cleanup, doc synchronization, and better compute access directly affect iteration speed even though they are not thesis contributions by themselves.

## Near-Term Next Steps
- *Current truth / active direction:* The agent scaffold now uses a thin-root, local-delta guide structure with repo-specific context split across narrower skills and an agent/tooling DB for scaffold debt.
- *Current truth / active direction:* Use discrete search, oracle-scored rollout diversity, and candidate-query Transformer `Q_H` as the first non-myopic learning milestone before attempting full continuous control.
- *Current truth / active direction:* Make bounded oracle-RRI lookahead versus one-step greedy under equal budget the first non-myopic comparison, then train `Q_H` from random-valid, oracle-greedy/lookahead, and oracle-scored temperature-softmax rollout data and evaluate it against one-step greedy/model scoring.
- *Current truth / active direction:* Use episode-normalized additive RRI or log-improvement reward for multi-step return while keeping current one-step RRI labels compatible with VIN training.
- *Current truth / active direction:* Scale within the current ecosystem before switching worlds: use small trusted subsets for correctness, then target the full 100 GT-mesh scenes / 4,608 snippet windows or a scene-level held-out subset with exact reporting of scenes, snippets, targets, trajectories, rollout seeds, transitions, and gaps.
- *Current truth / active direction:* Keep VIN improvements bounded and evidence-driven: ablate surface reconstruction, CORAL modifications, and auxiliary losses before committing to larger architectural rewrites.
- *Current truth / active direction:* Keep one oracle candidate-budget owner in candidate generation; the depth renderer now only caps already-pruned candidates at `max_candidates_final`.
- *Current truth / active direction:* Treat `rendering.unproject.backproject_depths_p3d_batch` as the canonical PyTorch3D depth-to-world unprojection path.
- *Current truth / active direction:* Keep immutable VIN offline stores lean by default: numeric blocks are canonical, and rich msgpack diagnostic DTO payloads are opt-in.
- *Current truth / active direction:* Empty oracle mesh crops are invalid inputs, not scene-level fallback labels.
- *Current truth / active direction:* Preserve the current geometry-first interpretation of counterfactual state while clarifying which missing modalities should be synthesized later.
- *Current truth / active direction:* Use GT-OBB-cropped target RRI as the V0 sanity/upper-bound metric, then move the main result to V1 observed/predicted OBB input with matched GT target-RRI labels under OBS-SEL / PRED-Q / GT-EVAL.
- *Current truth / active direction:* Use the Rerun offline inspector as a trust diagnostic for immutable offline-store samples. Inspector downsampling is display-only and must not mutate oracle labels, ranking metrics, or cached geometry.
- *Current truth / active direction:* Keep docs, Streamlit surfaces, and canonical memory aligned with code so advisor meetings stay grounded in the actual repo state.
- *Current truth / active direction:* Quarto docs keep retained QMD pages renderable while separating current thesis pages under `docs/contents/thesis/` from archived scratch/history under `docs/contents/archive/`; active work belongs in the TOML backlog rather than public TODO pages.
- *Current truth / active direction:* Literature/code knowledge graph work is routed through the `.agents/external/litkg-rs` submodule and the `semantic-scholar-litkg` skill, with ARIA-NBV-specific ingestion controlled by `.configs/litkg.toml` rather than toolkit hard-coding. The default representation is graphify-style durable Markdown/JSON plus Neo4j export bundles; the Neo4j export includes RustPython AST-backed `CodeFile`, `CodeModule`, `CodeSymbol`, `IMPORTS`, resolved local `CALLS`, and generated-context nodes from `make context` outputs. Agent entry points include `litkg kg find` and `litkg kg visualize` modality filters for code, docs, generated context, literature, memory, backlog, and external docs. CodeGraphContext is the optional live/deep code-symbol runtime; Graphiti and mempalace remain optional side integrations.
- *Current truth / active direction:* Project terminology is maintained in `docs/typst/shared/glossary.typ` and regenerated with `make glossary` into Quarto, Typst, KG-facing artifacts, and the Quarto glossary shortcode term map. QMD prose should use `{{< gls term-id >}}` or `{{< glsfull term-id >}}` instead of hand-written glossary links or inline redefinitions.

## Deferred but Important Extensions
- *Deferred extension:* Hestia-style hierarchical control that separates target selection from view realization and later introduces continuous motion.
- *Deferred extension:* IQL as a second offline-RL ablation, plus actor-critic or continuous-policy bridge design after Q_H is stable.
- *Deferred extension:* Entity-aware or task-aware RRI beyond the mandatory selected-target RRI protocol, such as richer weighted multi-entity objectives.
- *Deferred extension:* Counterfactual modality synthesis through Gaussian splats, synthetic SLAM, or world-model-style reconstruction for richer multi-step state.
- *Deferred extension:* Semantic-global planning with grounded subgoals such as portals, frontiers, entities, and regions layered on top of the geometric controller.
- *Deferred extension:* Real-device deployment, sim-to-real evaluation, and human-in-the-loop viewpoint guidance systems.

## Pointers
- Stable and already-adopted project choices live in `DECISIONS.md`.
- Advisor-pending scope calls and unresolved research/system questions live in `OPEN_QUESTIONS.md`.
- The raw archived idea scratchpad lives under `.agents/archive/docs/ideas.qmd`; public docs expose only the curated archive index at `docs/contents/archive/index.qmd`. Current thesis direction belongs in `docs/contents/thesis/roadmap.qmd` and `docs/contents/thesis/questions.qmd`.

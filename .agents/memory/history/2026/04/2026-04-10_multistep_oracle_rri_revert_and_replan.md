---
id: 2026-04-10_multistep_oracle_rri_revert_and_replan
date: 2026-04-10
title: "Revert Multi-Step Oracle-RRI Attempt And Replan"
status: done
topics: [counterfactuals, oracle-rri, data_handling, planning, memory]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/pose_generation/counterfactuals.py
  - .agents/memory/history/2026/04/2026-04-10_multistep_oracle_rri_revert_and_replan.md
artifacts:
  - snapshot branch `codex/multistep-oracle-rri-snapshot`
  - smaller issue decomposition for a future object-centric multistep pipeline
assumptions:
  - the current branch should keep the pre-existing counterfactual rollout and RL experiments, but discard the uncommitted oracle-RRI persistence attempt
  - future oracle-RRI rollouts will be object-centric rather than full-scene
---

Task

Preserve the discarded multistep oracle-RRI implementation attempt, restore the current branch to the pre-implementation counterfactual baseline, and record the findings and smaller follow-up packages for future GitHub issues.

Method

Created a preservation branch, `codex/multistep-oracle-rri-snapshot`, and archived the interrupted implementation there as a recovery bundle. The bundle contains the surviving source files, the two implementation debriefs that were accurate only for that attempt, and `.pyc` plus `pycdc` recovery artifacts for modules whose source had already been rolled back before the user asked to save the work. On the current branch, restored the original rollout/oracle-scoring baseline by rebuilding `pose_generation/counterfactuals.py` against the existing plotting, panel, and RL contracts, then replaced the misleading “feature landed” debriefs with this replan note.

Important findings

- The attempted data-handling integration was technically feasible with the existing optional `counterfactuals` record block and a geometry-first payload, but it coupled too much unstable runtime design into persistence too early.
- The current rollout runtime already supports structured candidate evaluation, cumulative metrics, and accumulated selected point clouds. That is enough surface area for planning and RL experiments without committing to offline-store contracts yet.
- The oracle scorer path has a real render-budget gotcha: `CandidateDepthRendererConfig.max_candidates_final` is tuned for dashboards, so any rollout or RL scorer that needs all valid candidates must align or override that budget explicitly.
- Policy-specific UI plumbing matters. A selection policy that semantically requires an evaluator cannot be represented as a bare enum on the dashboard without a paired scorer/runtime contract.
- The most important product change is conceptual rather than mechanical: future multistep oracle-RRI should be object-centric. Full-scene oracle RRI is the wrong long-term objective if planning should target objects of interest.
- The branching policy also needs a clearer contract. Expanding a tree with per-parent branching is not the right abstraction if the desired output is “exactly K distinct trajectories after horizon N.” The frontier should be trajectory-first and globally budgeted, not implicitly tree-shaped.

Discarded design choices from the attempt

- Do not persist multistep rollouts in the offline store yet.
- Do not freeze a serialized rollout wire format yet.
- Do not add a multistep training batch/DataModule path yet.
- Do not standardize a full-scene oracle-RRI rollout policy yet.
- Do not treat “beam width” as just post-expansion pruning of an exponentially growing tree without first deciding whether the target object is a bounded set of distinct chains.

Recommended workpackages

1. `spec(counterfactuals): formalize trajectory-first multistep rollout semantics`
   Goal: define the stable runtime contract before touching persistence.
   Scope: specify root/reference pose semantics, step payloads, termination rules, cumulative metrics, and the frontier policy for producing exactly `K` trajectories after horizon `N`.
   Acceptance: rollout APIs and plotting/RL surfaces use one agreed contract for “trajectory,” “branch,” “frontier,” and “selection score.”

2. `feat(objects): add object-of-interest selection for ATEK snippets`
   Goal: choose the planning target object per snippet.
   Scope: define how candidate objects are sourced, filtered, and selected for a snippet; record the selected object identity and geometry subset needed by downstream scoring.
   Acceptance: one snippet can be mapped to one explicit object target (or a controlled no-target outcome) through a deterministic, testable policy.

3. `feat(rri_metrics): implement object-centric oracle RRI`
   Goal: score only the object-relevant subset of the scene.
   Scope: adapt backprojection/filtering so only points belonging to the object of interest contribute to `P_t`, `P_q`, and the oracle objective; preserve the existing evaluator shape where possible.
   Acceptance: the scorer can compute object-scoped RRI with tests covering point filtering, empty-object cases, and metric consistency.

4. `feat(counterfactuals): integrate object-centric scoring into rollout selection`
   Goal: connect the stabilized rollout contract to the stabilized object-centric evaluator.
   Scope: add one or more explicit rollout selection policies over trajectory frontiers, compare greedy versus globally budgeted selection, and keep all logic in runtime/diagnostics only.
   Acceptance: counterfactual rollouts can produce `K` distinct trajectories under the chosen frontier policy without any offline-store dependency.

5. `feat(app,rl): align dashboard and RL diagnostics with the stabilized rollout/evaluator contract`
   Goal: make the existing inspection surfaces reflect the stabilized runtime rather than inventing their own policy handling.
   Scope: candidates panel policy controls, scorer wiring, render-budget alignment, and RL environment contract cleanup such as seed handling and reward-shell coverage.
   Acceptance: panel and RL diagnostics run on the stabilized rollout/evaluator stack with targeted tests for policy wiring and reward coverage.

6. `feat(data_handling): persist a minimal multistep payload only after runtime semantics settle`
   Goal: revisit offline persistence once the runtime contract is stable.
   Scope: persist only the agreed minimal trajectory payload through the existing record-block mechanism, document rebuild requirements, and keep training integration out of scope.
   Acceptance: stores can optionally carry minimal multistep payloads without changing one-step behavior or forcing schema churn from unresolved rollout semantics.

7. `feat(lightning): add multistep training batches after persisted payloads prove useful`
   Goal: expose multistep data to learning code only after the stored payload is validated.
   Scope: typed batch surface, dataset return format, and DataModule selection.
   Acceptance: multistep training data is opt-in, non-breaking, and built on the finalized persisted contract rather than on exploratory runtime objects.

Verification

- `cd aria_nbv && uv run ruff check aria_nbv/pose_generation/counterfactuals.py tests/pose_generation/test_counterfactuals.py tests/app/panels/test_candidates_panel.py tests/rl/test_counterfactual_env.py`
- `cd aria_nbv && uv run pytest --capture=no -vv tests/pose_generation/test_counterfactuals.py tests/app/panels/test_candidates_panel.py tests/rl/test_counterfactual_env.py`

Canonical state impact

No canonical state doc changed. This note records why the multistep oracle-RRI persistence attempt was discarded and how to restart it with a smaller, object-centric scope later.

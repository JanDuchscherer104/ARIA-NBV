---
id: 2026-05-18_ase_depth_rri_eval_contract
date: 2026-05-18
title: "ASE-Depth RRI Eval Contract"
status: done
topics: [rri, rollouts, depth, agents-db]
confidence: medium
canonical_updates_needed: []
files_touched:
  - .agents/issues.toml
  - .agents/todos.toml
  - aria_nbv/aria_nbv/rri_metrics/eval_pointclouds.py
  - aria_nbv/aria_nbv/rri_metrics/oracle_rri.py
  - aria_nbv/aria_nbv/pose_generation/counterfactuals.py
  - aria_nbv/aria_nbv/pose_generation/target_counterfactuals.py
---

## Task

Implemented the ASE-depth-root RRI evaluation contract from the dense-vs-semi-dense mismatch plan.

## Method

Added `ase_gt_depth_root` root-eval point clouds from observed-prefix `rgb/distance_m` frames, using EFM3D ray-distance unprojection and camera->rig->world transforms. Scene and target counterfactual scorers now default to oracle eval points rather than MPS semi-dense points and use root-normalized gain as the rollout score while retaining state-relative RRI diagnostics.

## Backlog

Added active RRI tracking under `issue-031` and `todo-079`, then amended M1, target-RRI, greedy/lookahead, rollout-retention, and Q_H training-view backlog rows so future scale work records eval-source lineage and reward semantics. The resolved LitKG `issue-029` remains preserved as historical memory.

## Verification

Initial targeted checks passed for the new eval point-cloud tests, rollout metric tests, and the target scorer fixture. Full targeted package, lint, and agents-db validation were still pending at debrief creation time.

## Canonical State Impact

`PROJECT_STATE.md` now records the separated actor-visible state and ASE-depth/root-normalized evaluation stream as current direction.

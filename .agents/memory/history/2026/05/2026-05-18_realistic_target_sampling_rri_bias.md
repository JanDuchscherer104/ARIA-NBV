---
id: 2026-05-18_realistic_target_sampling_rri_bias
date: 2026-05-18
title: "Realistic Target Sampling And RRI Bias Follow-Up"
status: done
topics: [target-selection, candidate-generation, rollouts, rri, agents-db]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/AGENTS_INTERNAL_DB.md
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/resolved.toml
  - aria_nbv/aria_nbv/data_handling/_target_selection.py
  - aria_nbv/aria_nbv/pose_generation/candidate_generation.py
  - aria_nbv/aria_nbv/pose_generation/candidate_generation_rules.py
  - aria_nbv/aria_nbv/pose_generation/candidate_mixture.py
  - aria_nbv/aria_nbv/pose_generation/counterfactuals.py
  - aria_nbv/aria_nbv/pose_generation/positional_sampling.py
  - aria_nbv/aria_nbv/pose_generation/types.py
---

## Task

Implemented the target-selection, realistic candidate-mixture, rollout-softmax, and agents-db parts of the target-selection sampling/RRI-bias plan.

## Method

Target selection now penalizes missing 2D projection instead of granting full visibility credit, uses effective semidense plus weighted EVL support, and records a GT match score distinct from IoU. Candidate generation now separates position families from view modes, makes the realistic V1 mixture local and target-aware, keeps `upper_bound_free_shell` explicit, and adds motion-realism diagnostics. Rollout temperature-softmax now defaults to median/IQR-normalized logits and supports optional sibling diversity guards for yaw, strategy, and target bearing.

## Backlog

Amended active TODOs for selector threshold audits, V1 matching/storage, realistic candidate mixtures, candidate-realism diagnostics, and stochastic rollout diversity. Resolved `todo-079` after validating the ASE-depth-root eval-stream implementation; kept `issue-031` open for downstream lineage propagation across target/rollout/storage/evidence surfaces.

## Verification

Passed targeted selector, candidate-mixture, RRI eval-pointcloud, rollout-metric, counterfactual, candidate-generation, pose-generation, orientation, and gravity-alignment tests. Touched Python files passed `uv run ruff check`.

## Canonical State Impact

No new canonical state file update is needed beyond the agents-db changes: `PROJECT_STATE.md` already records the ASE-depth-root/root-normalized RRI direction, and the current task refined implementation plus active follow-up records.

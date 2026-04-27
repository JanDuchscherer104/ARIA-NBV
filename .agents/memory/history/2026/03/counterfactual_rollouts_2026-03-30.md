---
id: 2026-03-30_counterfactual_rollouts
date: 2026-03-30
title: "Multi-step counterfactual pose rollouts"
status: done
topics: [pose-generation, counterfactuals, plotting, testing]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/pose_generation/counterfactuals.py
  - aria_nbv/aria_nbv/pose_generation/plotting.py
  - aria_nbv/aria_nbv/pose_generation/__init__.py
  - aria_nbv/tests/pose_generation/test_counterfactuals.py
artifacts:
  - multi-step counterfactual rollout generator with beam-style expansion
  - rollout plotting helpers for path and per-step shell inspection
assumptions:
  - counterfactual rollouts should reuse the existing one-step candidate generator rather than introducing a new motion model
---

Task

Implement multi-step counterfactual pose generation and plotting utilities grounded in the existing candidate-generation pipeline, after reviewing the relevant theory and outlook notes in the paper, slides, Hestia, and `ideas.qmd`.

Method

Added a new `CounterfactualPoseGenerator` config/runtime surface on top of the current one-step candidate generator. Each rollout step reuses the existing shell sampling, orientation, and rule pruning, then selects one or more valid candidates via built-in geometric policies or a caller-provided scorer. Extended `pose_generation.plotting` with rollout-aware builder methods and simple plotting helpers.

Findings

The main implementation subtlety was frame reuse across steps: recursively feeding selected candidate poses back into the one-step generator needs an explicit undo/reapply of the existing CW90 basis correction so multi-step expansion does not accumulate a spurious extra rotation. With that handled, the rollout layer stays compatible with the current candidate plotting and test surfaces.

Verification

- `ruff format aria_nbv/aria_nbv/pose_generation/counterfactuals.py aria_nbv/aria_nbv/pose_generation/plotting.py aria_nbv/aria_nbv/pose_generation/__init__.py aria_nbv/tests/pose_generation/test_counterfactuals.py`
- `ruff check aria_nbv/aria_nbv/pose_generation/counterfactuals.py aria_nbv/aria_nbv/pose_generation/plotting.py aria_nbv/aria_nbv/pose_generation/__init__.py aria_nbv/tests/pose_generation/test_counterfactuals.py`
- `cd aria_nbv && uv run pytest -s tests/pose_generation/test_counterfactuals.py`
- `cd aria_nbv && uv run pytest -s tests/pose_generation/test_plotting_helpers.py tests/pose_generation/test_pose_generation.py`

Canonical state impact

No canonical state doc changed. This adds an internal planning/diagnostic utility surface without changing the repo's top-level architecture claims.

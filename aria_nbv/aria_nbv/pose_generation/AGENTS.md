---
scope: module
applies_to: aria_nbv/aria_nbv/pose_generation/**
summary: Candidate generation, feasibility rule, orientation, and counterfactual pose guidance.
---

# Pose Generation Guidance

Follow [../../AGENTS.md](../../AGENTS.md) plus this file for work under
`aria_nbv/aria_nbv/pose_generation/`. Durable pose-generation notes live in
[README.md](README.md).

## Rules
- Preserve the distinction between physical rig-frame supervision and
  gravity-aligned or display-only sampling conveniences.
- Keep candidate feasibility rules explicit; do not silently relax invalid pose
  or collision handling.
- Use `PoseTW` and `CameraTW` at package boundaries.
- Keep plotting helpers separate from sampling and feasibility semantics.
- Counterfactual rollout helpers must state which modalities are logged,
  synthesized, or geometry-derived.

## Verification
- Run targeted pose-generation tests when sampling, feasibility, orientation, or
  counterfactual pose semantics change.
- Add deterministic tests for new rules where randomness or geometric thresholds
  affect candidate validity.

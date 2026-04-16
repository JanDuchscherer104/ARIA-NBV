---
scope: module
applies_to: aria_nbv/aria_nbv/rl/**
summary: Discrete-shell RL and counterfactual environment guidance.
---

# RL Guidance

Follow [../../AGENTS.md](../../AGENTS.md) plus this file for work under
`aria_nbv/aria_nbv/rl/`. Durable RL ownership notes live in [README.md](README.md).

## Rules
- Treat RL as an incremental planning scaffold, not proof of a finished
  end-to-end policy unless supported by explicit experiments.
- Keep observation, action, reward, termination, and invalid-action semantics
  typed and documented.
- Make the split between logged ego modalities and counterfactual or
  geometry-derived state explicit.
- Do not couple RL environments directly to Streamlit panels; panels should
  inspect environment outputs through typed helpers.

## Verification
- Run targeted RL environment tests for reset/step contracts, action validity,
  reward accounting, and deterministic seeds.
- Use Gymnasium/SB3 environment checks when environment interfaces change.

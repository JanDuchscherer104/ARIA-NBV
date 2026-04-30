---
name: counterfactual-rollout-planner
description: Use when ARIA-NBV work touches bounded counterfactual rollouts, non-myopic planning evaluation, invalid-action handling, Gymnasium/SB3 scaffolds, or the roadmap value/RL gate.
---

# Counterfactual Rollout Planner

## When To Use

Use this skill for:

- greedy, stochastic, beam, model-scored, or oracle rollouts
- cumulative RRI, path cost, invalid action rate, and runtime metrics
- Gymnasium/SB3 discrete-shell baselines
- M5/M6 planning or value/RL gate decisions

Do not use it for one-step VIN scoring unless the output drives rollout
selection or evaluation.

## Read First

1. `docs/contents/roadmap.qmd` sections M5 and M6
2. `docs/contents/questions.qmd` sections RQ5 and RQ6
3. `aria_nbv/AGENTS.md`
4. `.agents/memory/state/PROJECT_STATE.md`
5. Relevant `pose_generation` and `rl` tests

## Rules

- Keep horizons, branch factors, and beam widths explicit.
- Report cumulative target or scene RRI together with acquisition cost,
  invalid action rate, and runtime.
- Treat full continuous control and simulator-backed online RL as stretch work
  until the roadmap evidence gate passes.
- Keep actor-visible, critic-visible, and oracle-only signals separate.

## Verification

- `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py`
- `cd aria_nbv && uv run pytest tests/rl/test_counterfactual_env.py`
- `cd aria_nbv && uv run pytest tests/app/panels/test_rl_panel.py`
- `cd docs && quarto render contents/roadmap.qmd contents/questions.qmd` when claims or roadmap text change

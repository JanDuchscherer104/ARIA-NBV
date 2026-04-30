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

1. `docs/contents/thesis/roadmap.qmd` sections M5 and M6
2. `docs/contents/thesis/questions.qmd` sections RQ5 and RQ6
3. `aria_nbv/AGENTS.md`
4. `.agents/memory/state/PROJECT_STATE.md`
5. Relevant `pose_generation` and `rl` tests

## Rules

- Keep horizons, branch factors, and beam widths explicit.
- Treat `beam_width` as the number of sampled rollout chains retained per step,
  not as unbounded tree branching. The default bounded cost model is
  `O(B * L * N)` for beam width `B`, horizon `L`, and candidates per state `N`.
- For stochastic planning, score all valid candidates, sample with explicit
  softmax temperature, and use Gumbel-Top-k only when distinct sampled roots or
  chains are required.
- Score all candidates but materialize expensive counterfactual modalities only
  for selected actions or retained chains.
- Report cumulative target or scene RRI together with acquisition cost,
  invalid action rate, and runtime.
- Treat full continuous control and simulator-backed online RL as stretch work
  until the roadmap evidence gate passes.
- Keep actor-visible, critic-visible, and oracle-only signals separate.
- Rollout traces should record scene/snippet, horizon step, chain id, selected
  candidate id, score source, predicted RRI, oracle RRI when available,
  cumulative RRI, path cost, validity mask summary, invalid reason summary, and
  runtime.

## Verification

- `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py`
- `cd aria_nbv && uv run pytest tests/rl/test_counterfactual_env.py`
- `cd aria_nbv && uv run pytest tests/app/panels/test_rl_panel.py`
- `cd docs && quarto render contents/thesis/roadmap.qmd contents/thesis/questions.qmd` when claims or roadmap text change

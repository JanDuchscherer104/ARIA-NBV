---
name: counterfactual-rollout-planner
description: Use when ARIA-NBV work touches ASE counterfactual rollouts, non-myopic planning evaluation, invalid-action handling, stochastic branches, finite-candidate fitted Double-Q / Q_H, or the roadmap value/RL gate. Gymnasium/SB3 is post-M6 bridge work only.
metadata:
  applies_to:
    - "aria_nbv/aria_nbv/pose_generation/**"
    - "aria_nbv/aria_nbv/rl/**"
    - "docs/contents/thesis/**"
    - ".agents/references/rollout_zarr_q_invalidity_contract.md"
  triggers:
    - "counterfactual rollout"
    - "bounded lookahead"
    - "Q_H"
    - "invalid action"
  must_read:
    - "docs/contents/thesis/roadmap.qmd#roadmap-m5"
    - "docs/contents/thesis/questions.qmd#rq5-planning"
    - ".agents/memory/state/PROJECT_STATE.md"
    - ".agents/references/rollout_zarr_q_invalidity_contract.md"
  verification:
    - "cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py"
    - "cd aria_nbv && uv run pytest tests/rl/test_counterfactual_env.py"
---

# Counterfactual Rollout Planner

## When To Use

Use this skill for:

- deterministic oracle, greedy, stochastic, beam, model-scored, or oracle
  rollouts over ASE finite candidate sets
- cumulative RRI, path cost, invalid action rate, and runtime metrics
- finite-candidate fitted Double-Q / `Q_H` training and evaluation
- M5 planning/value decisions and M6 bridge boundaries

Do not use it for one-step VIN scoring unless the output drives rollout
selection or evaluation.

Use Gymnasium/SB3 only when the task explicitly targets the post-M6 online
simulator bridge after the ASE rollout and Q_H path is stable.

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
- Treat full continuous control and simulator-backed online RL as stretch or
  bridge work until the roadmap evidence gate passes.
- Keep actor-visible, critic-visible, and oracle-only signals separate.
- Mask invalid candidates before selection and preserve explicit invalid reason
  summaries.
- Oracle-evaluate selected learned rollouts before using them for claims.
- Log equal acquisition budget and candidate-budget parity for comparisons.
- Rollout traces should record scene/snippet, horizon step, chain id, selected
  candidate id, score source, predicted RRI, oracle RRI when available,
  cumulative RRI, path cost, validity mask summary, invalid reason summary, and
  runtime.

## Verification

- `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py`
- `cd aria_nbv && uv run pytest tests/rl/test_counterfactual_env.py`
- `cd aria_nbv && uv run pytest tests/app/panels/test_rl_panel.py`
- `cd docs && quarto render contents/thesis/roadmap.qmd contents/thesis/questions.qmd` when claims or roadmap text change

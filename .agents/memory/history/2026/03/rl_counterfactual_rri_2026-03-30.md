---
id: 2026-03-30_rl_counterfactual_rri
date: 2026-03-30
title: "Counterfactual RL scaffold and cumulative-RRI plotting"
status: done
topics: [rl, counterfactuals, plotting, rri, context7]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/OPEN_QUESTIONS.md
files_touched:
  - aria_nbv/aria_nbv/pose_generation/counterfactuals.py
  - aria_nbv/aria_nbv/pose_generation/plotting.py
  - aria_nbv/aria_nbv/pose_generation/__init__.py
  - aria_nbv/aria_nbv/rl/counterfactual_env.py
  - aria_nbv/aria_nbv/rl/__init__.py
  - aria_nbv/aria_nbv/app/panels/candidates.py
  - aria_nbv/pyproject.toml
  - .agents/references/context7_library_ids.md
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/OPEN_QUESTIONS.md
  - aria_nbv/tests/pose_generation/test_counterfactuals.py
  - aria_nbv/tests/rl/test_counterfactual_env.py
assumptions:
  - The first RL surface should stay discrete and shell-based instead of jumping directly to a continuous hierarchical policy.
---

Task
- Add a basic RL entrypoint following Hestia-style immediate rewards and extend counterfactual plotting to show cumulative-RRI-colored trajectories.

Method
- Reworked counterfactual evaluation from a raw score tensor into a structured evaluator result that can carry per-candidate metrics and selected candidate point clouds.
- Added an oracle-RRI-backed evaluator that reuses the existing candidate depth rendering, backprojection, and oracle RRI pipeline.
- Added a Gymnasium environment plus SB3 PPO factory for sequential shell-action selection with low-gamma defaults.
- Extended rollout plotting to color trajectories by cumulative RRI when available and updated the Context7 index with Gymnasium / SB3 references.

Findings
- The existing counterfactual rollout code was close to usable for RL once evaluator state was promoted into first-class rollout metadata.
- A shell-discrete environment was enough for SB3 `MultiInputPolicy` and kept the implementation aligned with the repo's current “planning / discrete first” outlook.
- SB3's env checker warns that `candidate_positions` and `history_positions` are unconventional non-flat observations; this is acceptable for the current `Dict` observation setup but is a sign that custom feature extractors or flattened variants may be worth trying next.

Verification
- `cd aria_nbv && uv sync --all-extras`
- `cd aria_nbv && ruff format aria_nbv/pose_generation/counterfactuals.py aria_nbv/pose_generation/plotting.py aria_nbv/pose_generation/__init__.py aria_nbv/rl/counterfactual_env.py aria_nbv/rl/__init__.py aria_nbv/app/panels/candidates.py tests/pose_generation/test_counterfactuals.py tests/rl/test_counterfactual_env.py`
- `cd aria_nbv && ruff check aria_nbv/pose_generation/counterfactuals.py aria_nbv/pose_generation/plotting.py aria_nbv/pose_generation/__init__.py aria_nbv/rl/counterfactual_env.py aria_nbv/rl/__init__.py aria_nbv/app/panels/candidates.py tests/pose_generation/test_counterfactuals.py tests/rl/test_counterfactual_env.py`
- `cd aria_nbv && uv run pytest -s tests/pose_generation/test_counterfactuals.py tests/pose_generation/test_plotting_helpers.py tests/rl/test_counterfactual_env.py tests/app/panels/test_candidates_panel.py`

Canonical State Impact
- `PROJECT_STATE.md` now records that multi-step counterfactual rollouts carry cumulative-RRI metrics and that a first Gymnasium/SB3 RL scaffold exists.
- `OPEN_QUESTIONS.md` now captures the remaining actor/critic state-design and discrete-vs-hierarchical action questions opened by the new RL scaffold.

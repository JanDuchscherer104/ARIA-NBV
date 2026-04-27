---
id: 2026-03-30_rl_streamlit_inspector
date: 2026-03-30
title: "Config-gated RL Streamlit inspector"
status: done
topics: [streamlit, rl, counterfactuals, app]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - aria_nbv/aria_nbv/app/config.py
  - aria_nbv/aria_nbv/app/app.py
  - aria_nbv/aria_nbv/app/panels/rl.py
  - aria_nbv/aria_nbv/app/panels/__init__.py
  - aria_nbv/aria_nbv/app/panels.py
  - aria_nbv/aria_nbv/rl/counterfactual_env.py
  - aria_nbv/tests/app/panels/test_rl_panel.py
assumptions:
  - Streamlit should stay evaluation-first; PPO training remains outside the app for now.
---

Task
- Add the second-phase RL-specific Streamlit surface while keeping additional features optional and config-driven.

Method
- Added `RlPageConfig` to the app config with feature gates for policy comparison, checkpoint playback, shell preview, step-shell plots, and episode tables.
- Added a new `RL Inspector` page that merges the active labeler settings into the RL env config, previews the initial shell, runs single episodes for `greedy_reward` / `random` / optional checkpoint PPO, and compares policies across seeded episode batches.
- Kept the panel thin by reusing the existing candidate and counterfactual plotting builders plus the new RL env helpers.

Findings
- Keeping the page evaluation-only makes the UI much simpler and avoids surprising background PPO work inside Streamlit reruns.
- Session-state caching keyed by `(sample, env config, policy, seed, checkpoint)` is enough to keep the page responsive without introducing new controller/state types.

Verification
- `cd aria_nbv && ruff format aria_nbv/app/config.py aria_nbv/app/app.py aria_nbv/app/panels/rl.py aria_nbv/app/panels/__init__.py aria_nbv/app/panels.py aria_nbv/rl/counterfactual_env.py tests/app/panels/test_rl_panel.py`
- `cd aria_nbv && ruff check aria_nbv/app/config.py aria_nbv/app/app.py aria_nbv/app/panels/rl.py aria_nbv/app/panels/__init__.py aria_nbv/app/panels.py aria_nbv/rl/counterfactual_env.py tests/app/panels/test_rl_panel.py`
- `cd aria_nbv && uv run pytest -s tests/app/panels/test_rl_panel.py tests/app/panels/test_candidates_panel.py tests/rl/test_counterfactual_env.py tests/pose_generation/test_counterfactuals.py`

Canonical State Impact
- `PROJECT_STATE.md` now records that the Streamlit app exposes an optional RL inspector page in addition to the underlying RL scaffold.

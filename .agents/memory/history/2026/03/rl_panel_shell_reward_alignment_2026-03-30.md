---
id: 2026-03-30_rl_panel_shell_reward_alignment
date: 2026-03-30
title: "RL Panel Shell/Reward Alignment Fix"
status: done
topics: [streamlit, rl, counterfactuals, oracle-rri]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/app/panels/rl.py
  - aria_nbv/aria_nbv/rl/counterfactual_env.py
  - aria_nbv/tests/app/panels/test_rl_panel.py
  - aria_nbv/tests/rl/test_counterfactual_env.py
---

Task
- Fix the RL Streamlit page crash where the default oracle-RRI evaluator returned fewer scores than the RL env's full valid shell.

Method
- Traced the mismatch to `CandidateDepthRendererConfig.max_candidates_final`, which remained at 60 while the RL env exposed the full oversampled shell as its discrete action space.
- Raised the merged RL-page reward-render budget to at least the shell capacity.
- Added env-side alignment for the default oracle reward path so non-Streamlit callers also get full-shell scoring.

Findings / outputs
- The RL page was exposing 114 shell actions while the default scorer rendered/scored only 60 of them.
- The env now raises the default oracle scorer depth budget to cover the shell it evaluates, and the page config mirrors that behavior.

Verification
- `cd /home/jandu/repos/NBV/aria_nbv && ruff check aria_nbv/rl/counterfactual_env.py aria_nbv/app/panels/rl.py tests/app/panels/test_rl_panel.py tests/rl/test_counterfactual_env.py`
- `cd /home/jandu/repos/NBV/aria_nbv && uv run pytest -s tests/app/panels/test_rl_panel.py tests/rl/test_counterfactual_env.py`

Canonical state impact
- None.

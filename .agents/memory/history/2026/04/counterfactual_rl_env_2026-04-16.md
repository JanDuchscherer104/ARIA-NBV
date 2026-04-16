---
id: 2026-04-16_counterfactual_rl_env
date: 2026-04-16
title: "Add counterfactual RL environment scaffold"
status: done
topics: [rl, gymnasium, stable-baselines3]
confidence: high
canonical_updates_needed: []
---

## Verification

- `/home/jandu/repos/NBV/aria_nbv/.venv/bin/python -m ruff check aria_nbv/aria_nbv/rl/__init__.py aria_nbv/aria_nbv/rl/counterfactual_env.py aria_nbv/tests/rl/test_counterfactual_env.py`
- `/home/jandu/repos/NBV/aria_nbv/.venv/bin/python -m pytest -s aria_nbv/tests/rl/test_counterfactual_env.py`

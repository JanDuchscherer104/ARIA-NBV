---
id: 2026-04-16_counterfactual_rollout_core
date: 2026-04-16
title: "Add counterfactual rollout core"
status: done
topics: [pose-generation, counterfactuals, plotting]
confidence: high
canonical_updates_needed: []
---

## Verification

- `/home/jandu/repos/NBV/aria_nbv/.venv/bin/python -m ruff check aria_nbv/aria_nbv/pose_generation/__init__.py aria_nbv/aria_nbv/pose_generation/candidate_generation.py aria_nbv/aria_nbv/pose_generation/plotting.py aria_nbv/aria_nbv/pose_generation/utils.py aria_nbv/aria_nbv/pose_generation/counterfactuals.py aria_nbv/tests/pose_generation/test_counterfactuals.py`
- `/home/jandu/repos/NBV/aria_nbv/.venv/bin/python -m pytest -s aria_nbv/tests/pose_generation/test_counterfactuals.py`

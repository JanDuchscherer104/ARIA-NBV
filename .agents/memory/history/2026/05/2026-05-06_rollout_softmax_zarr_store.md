---
id: 2026-05-06_rollout_softmax_zarr_store
date: 2026-05-06
title: "Rollout Softmax Zarr Store"
status: done
topics: [counterfactual-rollout, zarr, q-h, memory, litkg]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - aria_nbv/aria_nbv/pose_generation/counterfactuals.py
  - aria_nbv/aria_nbv/pose_generation/rollout_trace.py
  - aria_nbv/aria_nbv/pose_generation/__init__.py
  - aria_nbv/aria_nbv/data_handling/_rollout_zarr_store.py
  - aria_nbv/aria_nbv/data_handling/__init__.py
  - aria_nbv/tests/pose_generation/test_counterfactuals.py
  - aria_nbv/tests/data_handling/test_rollout_zarr_store.py
  - .agents/work/research-review-proposal-distillation/rollout-qh-zarr-dto-digest.md
  - .agents/memory/state/DECISIONS.md
---

## Task

Implemented the first selected-action rollout replay path for finite-candidate
`Q_H`: masked temperature-softmax selection, provenance-rich rollout traces, and
a standalone `rollouts.zarr` smoke writer/reader/validator.

## Method

The counterfactual selector now supports explicit `random_valid`,
`oracle_greedy`, and `temperature_softmax` policy names while preserving the
existing policies. The trace DTO records policy, score source, temperature,
logits, probabilities, log-probabilities, entropy, selected log-probability,
RNG seed, transition id, and invalid reason bitsets. The rollout Zarr writer
materializes full-shell candidate rows, selected-action TD fields, and padded
`q_h` arrays with dense all-action oracle-Q targets left unavailable and `NaN`.

## Verification

- `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py`
- `cd aria_nbv && uv run pytest tests/data_handling/test_rollout_zarr_store.py`
- `cd aria_nbv && uv run pytest tests/data_handling/test_public_api_contract.py`
- `cd aria_nbv && uv run ruff check aria_nbv/pose_generation/counterfactuals.py aria_nbv/pose_generation/rollout_trace.py aria_nbv/pose_generation/__init__.py aria_nbv/data_handling/_rollout_zarr_store.py aria_nbv/data_handling/__init__.py tests/pose_generation/test_counterfactuals.py tests/data_handling/test_rollout_zarr_store.py`

## Canonical State Impact

`DECISIONS.md` now records that `rollouts.zarr` is standalone, masked
temperature-softmax is the first stochastic data-diversity policy, selected
transition replay is first, and dense all-action oracle-Q materialization is
later work.

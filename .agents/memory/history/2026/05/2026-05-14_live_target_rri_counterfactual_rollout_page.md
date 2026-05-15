---
id: 2026-05-14_live_target_rri_counterfactual_rollout_page
date: 2026-05-14
title: "Live Target-RRI Counterfactual Rollout Page"
status: done
topics: [streamlit, rollouts, target-rri, inspection]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/app/app.py
  - aria_nbv/aria_nbv/app/config.py
  - aria_nbv/aria_nbv/app/panels/counterfactual_rollouts.py
  - aria_nbv/aria_nbv/app/panels/candidates.py
  - aria_nbv/tests/app/panels/test_counterfactual_rollouts_panel.py
---

## Task

Added a dedicated Streamlit Counterfactual Rollouts page so live multi-step
rollouts are generated from `VinOfflineSample` roots with target-specific RRI
as the default scoring mode. Candidate Poses now stays focused on one-step
candidate diagnostics; stored `rollouts.zarr` inspection moved to the new page.

## Method

The new page loads a VIN offline store in `sample` mode with live snippet,
GT mesh, detected OBBs, GT OBBs, and backbone fields. It exposes target
selection, target-match status, target-aware candidate mixtures, rollout
controls, target/scorer controls, and stored rollout Rerun launch actions.
Geometry mode is explicit and does not fabricate cumulative RRI. Scene RRI is
available as a diagnostic mode. The bridge RL page is hidden by default.

Follow-up fix on 2026-05-14: the live rollout page initially defaulted to CUDA
when Torch reported a visible GPU, but the local PyTorch3D rasterizer is not
compiled with GPU support. The page now exposes only CPU for live rollout
rendering, and the target/scene depth scorer config receives the same explicit
CPU device as candidate generation.

## Verification

- `cd aria_nbv && uv run ruff check aria_nbv/app aria_nbv/pose_generation tests/app tests/test_panels_dispatcher.py`
- `cd aria_nbv && uv run pytest tests/app/panels -q`
- `cd aria_nbv && uv run pytest tests/data_handling/test_target_selection.py tests/pose_generation/test_counterfactuals.py tests/rl/test_counterfactual_env.py -q`
- `cd aria_nbv && uv run pytest tests/app/panels tests/data_handling/test_target_selection.py tests/pose_generation/test_counterfactuals.py -q`
- Headless Chrome/CDP loaded `/_page_counterfactual_rollouts`, clicked
  `Load sample and targets`, clicked `Run / refresh live rollouts`, and
  verified a rendered `target_rri` rollout table without the PyTorch3D GPU
  support error. Screenshot artifact:
  `/tmp/counterfactual_rollouts_after_run.png`.
- scoped `git diff --check` passed for touched tracked files and untracked new files

Full `git diff --check` is still blocked by pre-existing dirty binary PDF
changes in `docs/typst/thesis/advisor_distillation.pdf`.

## Canonical State Impact

No canonical state updates are required. This is an app workflow correction:
target-RRI live rollout inspection now has a first-class UI surface instead of
being hidden under the Candidate Poses page.

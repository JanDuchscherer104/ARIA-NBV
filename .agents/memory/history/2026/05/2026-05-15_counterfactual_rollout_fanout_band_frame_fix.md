---
id: 2026-05-15_counterfactual_rollout_fanout_band_frame_fix
date: 2026-05-15
title: "Counterfactual Rollout Fanout Band And Frame Fix"
status: done
topics: [streamlit, rollouts, plotting, frames]
confidence: high
canonical_updates_needed: []
---

## Task

Implemented the live Counterfactual Rollouts display fix requested on
2026-05-15: replace min/mean/max fanout lines with an empirical central 95%
valid-candidate target-RRI band, and remove the extra CW90 rotation that made
counterfactual frusta appear bottom-up.

## Outputs

- The live dashboard now plots selected target RRI against the 2.5-97.5
  percentile range of finite valid-candidate target RRI per rollout step.
- `CounterfactualStepResult.selected_pose_world` now returns the selected raw
  valid candidate pose; trajectory pose chains and regenerated rollout roots no
  longer inherit a second display rotation.
- Counterfactual selected-frusta and step-shell frusta now render from raw
  rollout candidate poses. Normal candidate-page and Data-page display
  conventions were left unchanged.

## Verification

- `cd aria_nbv && uv run pytest tests/app/panels/test_counterfactual_rollouts_panel.py tests/pose_generation/test_counterfactuals.py -q`
- `cd aria_nbv && uv run pytest tests/app/panels tests/pose_generation/test_counterfactuals.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/app aria_nbv/pose_generation tests/app tests/pose_generation`

## Notes

Existing persisted rollout stores generated before this fix may need
regeneration for geometry-correct rollout frustum inspection. No rollout-Zarr
schema change was made.

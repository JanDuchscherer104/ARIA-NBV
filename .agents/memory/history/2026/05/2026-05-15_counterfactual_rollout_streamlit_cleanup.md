---
id: 2026-05-15_counterfactual_rollout_streamlit_cleanup
date: 2026-05-15
title: "Counterfactual Rollout Streamlit Cleanup"
status: done
topics: [streamlit, rollouts, rerun, rri, diagnostics]
confidence: high
canonical_updates_needed: []
---

## Task

Implemented the Counterfactual Rollouts Streamlit cleanup plan: live rollout
generation stays on the Counterfactual Rollouts page, persisted rollout-Zarr
inspection moved to the VIN Offline Dataset page, Rerun launch now has native
and web-viewer paths, and rollout metric semantics live under `rri_metrics`.

## Outputs

- Added target-rollout metric helpers for selected `G_t^(H)`, endpoint
  `J_e^(H)`, and log-gain from target point-mesh before/after fields.
- Added live Plotly branch-summary dashboards for selected target RRI,
  cumulative return, endpoint metrics when available, candidate fanout bands,
  and top-k candidate target RRI.
- Added stored rollout dashboards and decoded candidate strategy/mixture labels
  on the VIN Offline Dataset page.
- Updated Rerun web serving command construction for Rerun 0.26 by mapping the
  app's proxy-port option to `rerun --port`.
- Added actor-visible target semidense crop overlays and target-RRI-colored,
  display-rotated rollout frusta.

## Verification

- `cd aria_nbv && uv run pytest tests/app/panels tests/rerun_inspector tests/rri_metrics tests/pose_generation/test_counterfactuals.py -q`
- `cd aria_nbv && uv run pytest tests/pose_generation -q`
- `cd aria_nbv && uv run ruff check aria_nbv/app aria_nbv/rerun_inspector aria_nbv/rri_metrics aria_nbv/pose_generation tests/app tests/rerun_inspector tests/rri_metrics tests/pose_generation`

## Notes

The rollout store schema was intentionally left unchanged. Stored rollout
endpoint/log-gain plots remain unavailable until the store persists target
point-mesh before/after fields.

---
id: 2026-05-06_rollout_zarr_stress_diagnostic
date: 2026-05-06
title: "Rollout Zarr Stress Diagnostic"
status: done
topics: [diagnostics, rollout, zarr, q-h]
confidence: high
canonical_updates_needed: []
artifacts:
  - .artifacts/rollout_zarr_stress/2026-05-06_multistep_softmax_zarr/rollouts.zarr
  - .artifacts/rollout_zarr_stress/2026-05-06_multistep_softmax_zarr/structural_summary.json
  - .artifacts/rollout_zarr_stress/2026-05-06_multistep_softmax_zarr/mask_probability_heatmaps.png
  - .artifacts/rollout_zarr_stress/2026-05-06_multistep_softmax_zarr/selected_paths_xy.png
---

## Task

Stress-tested the newly implemented standalone rollout Zarr replay path by
writing a deterministic synthetic multi-step store with greedy, random-valid,
and temperature-softmax policies, invalid full-shell rows, and selected-action
TD links.

## Findings

The final artifact contains 6 rollout chains, 18 persisted steps, 864 full-shell
candidate rows, 454 invalid candidate rows, and 12 nonterminal selected-action
TD links. Structural validation passed; invalid candidates have zero
probability and no finite logits, probability mass over valid candidates sums
to one within float tolerance, `q_train_mask` implies `valid_action_mask`, and
dense all-action oracle-Q tensors remain unavailable/NaN as intended.

The first attempted stress extent was too tight and terminated after one step;
the final artifact uses a wider synthetic free-space extent so the store
exercises multi-step transitions and invalid rows together.

## Verification

- Generated and validated `.artifacts/rollout_zarr_stress/2026-05-06_multistep_softmax_zarr/rollouts.zarr`.
- Inspected Zarr group/array shapes and root attrs from the written store.
- Rendered and visually inspected mask/probability heatmaps and selected-path
  plots from persisted candidate rows.

## Canonical State Impact

No canonical memory update was needed. The diagnostic confirmed the existing
selected-action replay and dense-oracle-Q-later decisions.

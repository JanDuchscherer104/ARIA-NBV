---
id: 2026-05-15_efm_semantic_class_name_resolution
date: 2026-05-15
title: "EFM Semantic Class Name Resolution"
status: done
topics: [data-handling, target-selection, rerun, streamlit]
confidence: high
canonical_updates_needed: []
---

## Task

Fixed target and OBB class-name resolution for EFM semantic ids. The visible
symptom was an active rollout target label showing `28` instead of `window`.

## Method

Canonicalized semantic-name metadata as sparse `dict[int, str]` maps across
offline-store decode/write, `EvlBackboneOutput`, target selection, Rerun OBB
labels, and Counterfactual Rollouts help text. Added a shared semantic-name
helper so UI and diagnostics do not keep separate list-index assumptions.

## Findings

The existing smoke store already contained the correct raw EFM payload
`{"28": "window"}`. The bug was introduced during typed decode and compact
normalization, which converted sparse semantic ids into dense or identity-name
lists.

## Verification

- `cd aria_nbv && uv run pytest tests/data_handling/test_target_selection.py tests/data_handling/test_vin_offline_store.py tests/rerun_inspector/test_loggers.py tests/app/panels/test_counterfactual_rollouts_panel.py -q`
- `cd aria_nbv && uv run pytest tests/lightning/test_vin_batch_collate.py tests/pose_generation/test_counterfactuals.py -q`
- real-data smoke on `.data/offline_cache/vin_offline` sample 0 confirmed
  `target 0 · window · sem=28 inst=51297`.

## Canonical State Impact

No additional canonical state update is needed. Sparse semantic-name maps are
now enforced directly by tests on the active data path.

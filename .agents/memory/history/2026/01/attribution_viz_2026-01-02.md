---
id: 2026-01-02_attribution_viz_2026-01-02
date: 2026-01-02
title: "Attribution Viz 2026 01 02"
status: legacy-imported
topics: [attribution, viz, 2026, 01, 02]
source_legacy_path: ".codex/attribution_viz_2026-01-02.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Testing & Attribution: richer VIN attributions

## Summary
- Added pose-vector attribution mode to the Testing & Attribution panel using VIN's pose encoder submodules.
- Added group ablation diagnostics (pose_enc/global_feat/extra) with full/ablated/only scores and deltas.
- Switched attribution display to use raw Captum attributions with optional abs + normalization toggles.
- Added `infer_pose_vec_groups` helper for semantic pose-vector grouping and tests.

## Files changed
- `oracle_rri/oracle_rri/app/panels/testing_attribution.py`
- `oracle_rri/oracle_rri/vin/pose_encoders.py`
- `tests/vin/test_pose_vec_groups.py`

## Tests
- `python -m pytest tests/vin/test_pose_vec_groups.py`
- `python -m pytest tests/vin/test_vin_model_v2_integration.py -m integration`

## Notes / follow-ups
- Pose-vector attribution relies on LFF or SH submodules; if new pose encoders are added, update the adapter.
- For deeper spatial interpretability, consider adding optional attention-weight or voxel-channel attribution visualizations in VIN v2.

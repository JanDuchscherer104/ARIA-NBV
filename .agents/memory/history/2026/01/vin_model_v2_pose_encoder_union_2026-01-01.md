---
id: 2026-01-01_vin_model_v2_pose_encoder_union_2026-01-01
date: 2026-01-01
title: "Vin Model V2 Pose Encoder Union 2026 01 01"
status: legacy-imported
topics: [model, v2, pose, encoder, union]
source_legacy_path: ".codex/vin_model_v2_pose_encoder_union_2026-01-01.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Task: VinModelV2 pose encoder discriminated union (2026-01-01)

## Changes
- Added `oracle_rri/oracle_rri/vin/pose_encoders.py` with a pose-encoder interface, shared output dataclass, and encoder variants:
  - `R6dLffPoseEncoder` (default, learns translation/rotation scaling)
  - `ShellLffPoseEncoder` (shell descriptor + LFF)
  - `ShellShPoseEncoderAdapter` (shell descriptor + SH)
- `VinModelV2Config` now uses a discriminated-union `pose_encoder` config (`kind` field) and adds `pos_grid_encoder_lff` for LFF positional keys on XYZ.
- `VinModelV2` delegates pose encoding to the encoder class; `_encode_pose_r6d` and pose scaling logic removed from the model.
- Global pooling keys now apply LFF to normalized voxel XYZ before the positional projection.
- Docs updated in `docs/contents/impl/vin_nbv.qmd` to reflect configurable pose encoder, roll note, and LFF pos-grid keys.

## Roll note
- Shell-based encoders use only the forward direction and ignore roll about the forward axis. This is documented in both code and docs; acceptable when roll jitter is small.

## Tests
- `pytest oracle_rri/tests/vin/test_vin_model_v2_gradients.py` failed during collection due to missing `seaborn` (imported via `oracle_rri.vin.plotting`).
- `pytest tests/vin/test_vin_model_v2_integration.py::test_vin_v2_forward_on_real_snippet_cpu` failed during collection due to missing `power_spherical`.

## Suggestions / follow-ups
- Install missing deps (`seaborn`, `power_spherical`) in the active environment before re-running tests.
- If SH-based encodings become important, consider adding a roll-sensitive scalar (e.g., dot with reference up/right) or switch to `r6d_lff`.

## Runtime fix (pose_encoder_lff shadowing)
- Issue: `PoseEncoder` defined `pose_encoder_lff = None` at class level, which shadowed nn.Module’s registered submodule. Accessing `enc.pose_encoder_lff` returned `None`, causing `out_dim` to fail.
- Fix: removed the class-level default and kept only the type annotation so Module attribute lookup resolves to `_modules`.

## Default sizing adjustments
- Reduced default pose LFF size in `R6dLffPoseEncoderConfig` to fourier_dim=64, hidden_dim=128, output_dim=32.
- Reduced default pos-grid LFF size in `VinModelV2Config` to fourier_dim=32, hidden_dim=32, output_dim=16.

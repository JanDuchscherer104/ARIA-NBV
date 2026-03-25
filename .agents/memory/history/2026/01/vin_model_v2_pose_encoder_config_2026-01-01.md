---
id: 2026-01-01_vin_model_v2_pose_encoder_config_2026-01-01
date: 2026-01-01
title: "Vin Model V2 Pose Encoder Config 2026 01 01"
status: legacy-imported
topics: [model, v2, pose, encoder, config]
source_legacy_path: ".codex/vin_model_v2_pose_encoder_config_2026-01-01.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Task: VinModelV2 pose encoder configurability (2026-01-01)

## Findings
- `oracle_rri/oracle_rri/vin/model_v2.py` hardcodes LFF on `[t, r6d]` and keeps `_encode_pose_r6d` inside the model.
- `oracle_rri/oracle_rri/vin/model_v1_SH.py` encodes shell descriptors `(u, f, r, s)` with `ShellShPoseEncoder`; forward direction is `f = R_rig_ref_cam * z_cam`.
- `oracle_rri/oracle_rri/vin/model.py` already supports `pose_encoding_mode` (`shell_lff` vs `t_r6d_lff`) with shared LFF config.
- `oracle_rri/oracle_rri/vin/spherical_encoding.py` expects `(u, f, r, scalars)`; roll is not represented unless extra scalars are added.

## Suggestions
- Introduce a pose-encoder interface + output dataclass (e.g., `PoseEncodingOutput`) and move pose-encoding logic out of `VinModelV2`.
  - Implement `R6dLffPoseEncoder` (includes learnable scale params) and a shell wrapper (`ShellLffPoseEncoder` and/or `ShellShPoseEncoderAdapter`).
- Add a configurable `pose_encoding_mode` (or a discriminated union config) in `VinModelV2Config` to select the encoder.
- For SH orientation: `f = R_rig_ref_cam * z_cam` is correct for the forward axis in LUF; add roll awareness only if roll jitter matters.
  - If roll matters, prefer LFF on `r6d` (full SO(3)) or add an additional direction (up/right) or scalar roll features.
- Keep voxel positional keys as XYZ (current `pos_proj`) or LFF; SH for voxel positions is only justified if you explicitly convert to `(u, r)` and accept the spherical bias.

## Open questions
- Do we want roll encoded in SH mode (via extra direction/scalars) or is forward-only sufficient?
- Should voxel positional keys stay in XYZ, or should we experiment with SH/LFF for the pos grid?

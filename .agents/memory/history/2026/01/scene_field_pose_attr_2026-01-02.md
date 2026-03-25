---
id: 2026-01-02_scene_field_pose_attr_2026-01-02
date: 2026-01-02
title: "Scene Field Pose Attr 2026 01 02"
status: legacy-imported
topics: [scene, field, pose, attr, 2026]
source_legacy_path: ".codex/scene_field_pose_attr_2026-01-02.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Scene-field + pose input attribution enhancements

## Summary
- Added scene-field channel attribution mode in Testing & Attribution panel, including per-channel scores and spatial heatmap (depth-projected).
- Added pose-dim bar chart for pose input attribution.
- Scene-field attribution uses VIN v2 field_proj + global_pooler with fixed pose_enc/pos_grid to backprop to `field_in` (scene_field_channels).

## Files changed
- `oracle_rri/oracle_rri/app/panels/testing_attribution.py`

## Tests
- `ruff check oracle_rri/oracle_rri/app/panels/testing_attribution.py`

## Notes
- Scene-field attribution requires VIN v2 (needs `_pos_grid_from_pts_world` and `field_in` debug).
- Heatmaps are mean/max projections over depth for the selected channel.

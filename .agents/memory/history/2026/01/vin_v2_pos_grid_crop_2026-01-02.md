---
id: 2026-01-02_vin_v2_pos_grid_crop_2026-01-02
date: 2026-01-02
title: "Vin V2 Pos Grid Crop 2026 01 02"
status: legacy-imported
topics: [v2, pos, grid, crop, 2026]
source_legacy_path: ".codex/vin_v2_pos_grid_crop_2026-01-02.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

## Summary

- Fixed VIN v2 positional grid construction to handle padded voxel-center grids by center-cropping `pts_world` to the field grid size before normalization.
- Added a focused unit test to cover the padded-grid path and ensure expected cropping behavior.

## Root Cause

- `VinModelV2` builds `field` with a valid-kernel `Conv3d` (`kernel_size=3`, no padding), shrinking the voxel grid by 2 in each axis.
- `voxel/pts_world` remains at the pre-conv resolution, so `pos_grid_from_pts_world` rejected the mismatch (`48^3` vs `46^3`).

## Fix Details

- `oracle_rri/oracle_rri/vin/vin_v2_utils.py`:
  - infer padded grid shapes for `pts_world` (flat or 5D),
  - center-crop to match the field grid shape,
  - preserve strict validation when the mismatch cannot be resolved.
- `tests/vin/test_vin_v2_utils.py`: new unit test validating center-crop behavior.

## Tests Run

- `oracle_rri/.venv/bin/python -m pytest tests/vin/test_vin_v2_utils.py tests/vin/test_vin_model_v2_integration.py`

## Follow-ups / Suggestions

- Consider documenting the `Conv3d` shrink in VIN v2 notes to prevent future shape mismatches.

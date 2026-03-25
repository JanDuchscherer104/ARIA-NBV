---
id: 2026-01-02_vin_geometry_cw90_plotting_2026-01-02
date: 2026-01-02
title: "Vin Geometry Cw90 Plotting 2026 01 02"
status: legacy-imported
topics: [geometry, cw90, plotting, 2026, 01]
source_legacy_path: ".codex/vin_geometry_cw90_plotting_2026-01-02.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VIN geometry plotting cw90 alignment

## Summary
- Added a per-frusta display rotation toggle in `SnippetPlotBuilder.add_frusta` and wired VIN geometry to disable UI rotation for frusta.
- Geometry panel now undoes cw90 on reference/candidate poses when `apply_cw90_correction=True`, keeping voxel grid + backbone evidence aligned in raw world coordinates.
- Candidate centers in `build_geometry_overview_figure` now derive from the reference/candidate poses when available, ensuring display consistency with the plotted pose frame.

## Key changes
- `oracle_rri/oracle_rri/data/plotting.py`: `add_frusta(..., is_rotate_yaw_cw90=True)` added; rotation is conditional.
- `oracle_rri/oracle_rri/vin/plotting.py`: helpers to batch/broadcast poses; candidate centers can be computed from reference + candidate poses; `build_geometry_overview_figure` gets `frustum_rotate_yaw_cw90` and passes it to `add_frusta`.
- `oracle_rri/oracle_rri/app/panels/vin_diag_tabs/geometry.py`: when VIN v2 `apply_cw90_correction=True`, undo cw90 for plotting poses; frusta rotation disabled for raw-world display.

## Potential issues / open questions
- If `apply_cw90_correction=False`, VIN debug outputs and candidate poses remain in the UI-rotated frame, but the voxel grid/semidense points are still raw-world; geometry can still look inconsistent. Consider enforcing raw-world display or adding a full “UI-rotated display” toggle that rotates **all** scene elements (mesh, semidense, voxel, backbone points).

## Suggestions
- Add a UI toggle to explicitly select display frame (raw world vs UI-rotated), with a clear warning about consistency.
- Centralize cw90 display handling in plotting utils to avoid reintroducing mixed-frame bugs in new panels.

## 2026-01-02 follow-up
- Fixed VIN geometry panel to undo cw90 when `apply_cw90_correction=True` (previously re-applied cw90), which caused yaw mismatch between frusta/candidates and voxel grid in raw world.

## 2026-01-02 UI display rotation
- Added `display_rotate_yaw_cw90` to VIN geometry plotting to rotate the voxel frame and backbone evidence into UI display space.
- Evidence points are now rotated around the voxel frame origin to stay aligned with the rotated voxel grid.
- Geometry tab now enables UI display rotation and passes the flag through.

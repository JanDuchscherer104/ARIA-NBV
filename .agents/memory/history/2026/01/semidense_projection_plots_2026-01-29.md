---
id: 2026-01-29_semidense_projection_plots_2026-01-29
date: 2026-01-29
title: "Semidense Projection Plots 2026 01 29"
status: legacy-imported
topics: [semidense, projection, plots, 2026, 01]
source_legacy_path: ".codex/semidense_projection_plots_2026-01-29.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Semidense Projection Plots + Formulas (2026-01-29)

## Scope
Add the provided projection plots to the VINv3 semidense slides and include explicit formulas for:
- camera-plane projection (PerspectiveCameras),
- grid binning / aggregation,
- reliability-weight normalization (obs_count + inv_dist_std).

## Files touched
- `docs/typst/slides/slides_4.typ`

## What changed
- Inserted a new slide **“Semidense projection: transforms + grid maps”** directly after Branch 6:
  - Left column: explicit projection equations, valid mask, binning, per-cell map formulas, and reliability-weight normalization.
  - Right column: a 3-panel figure showing the requested plots:
    - `docs/figures/app-paper/semi-dense-counts-proj.png`
    - `docs/figures/app-paper/semi-dense-weight-proj.png`
    - `docs/figures/app-paper/semi-dense-std-proj.png`
  - Added the aggregation intuition for the current `offline_only` config:
    - `semidense_proj_grid_size = 12` (from `.configs/offline_only.toml`)
    - If `image_size = 120x120`, each grid cell corresponds to ~10x10 px.

## Build check
- `typst compile typst/slides/slides_4.typ /tmp/slides_4.pdf --root .` (from `docs/`) succeeded.

## Notes / follow-ups
- The slide uses the same validity mask and grid binning as `VinModelV3._encode_semidense_projection_features` and the plotting helpers in `oracle_rri/vin/plotting.py`.
- If future runs change image_size or grid size, update the “aggregation intuition” line accordingly.

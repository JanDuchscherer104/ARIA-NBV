---
id: 2026-01-24_vin_v3_fletcher_diagrams_2026-01-24
date: 2026-01-24
title: "Vin V3 Fletcher Diagrams 2026 01 24"
status: legacy-imported
topics: [v3, fletcher, diagrams, 2026, 01]
source_legacy_path: ".codex/vin_v3_fletcher_diagrams_2026-01-24.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Task Notes: VINv3 Fletcher Diagrams (2026-01-24)

## Scope
- Created a set of modular Fletcher diagrams for `oracle_rri/oracle_rri/vin/model_v3.py`.
- Diagrams are intentionally separated so they can be chained in a future composite.

## Outputs
- `docs/typst/diagrams/vin_v3/vin_v3_style.typ`
- `docs/typst/diagrams/vin_v3/vin_v3_01_inputs.typ`
- `docs/typst/diagrams/vin_v3/vin_v3_02_scene_field.typ`
- `docs/typst/diagrams/vin_v3/vin_v3_03_pose_global.typ`
- `docs/typst/diagrams/vin_v3/vin_v3_04_projection_film.typ`
- `docs/typst/diagrams/vin_v3/vin_v3_05_head.typ`
- Rendered previews: `docs/typst/diagrams/vin_v3/_render/*.png`

## Diagram Coverage
1) Inputs + PrepareInputs stage.
2) Scene-field channel derivation + projection.
3) Pose encoding + global pooling + voxel validity gating.
4) Voxel + semidense projection stats with FiLM modulation.
5) Head: concat, MLP, CORAL, candidate_valid mask.

## Notes
- Labels are kept short for legibility; arrows encode the main data flow.
- Layout is sized for future chaining (horizontal flow, minimal vertical clutter).

## Tests
- `typst compile` was used to render PNG previews (requires elevated permissions because Typst runs via snap).

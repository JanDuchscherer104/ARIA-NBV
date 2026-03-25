---
id: 2026-01-29_voxel_branch_doc_alignment_2026-01-29
date: 2026-01-29
title: "Voxel Branch Doc Alignment 2026 01 29"
status: legacy-imported
topics: [voxel, branch, doc, alignment, 2026]
source_legacy_path: ".codex/voxel_branch_doc_alignment_2026-01-29.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Voxel-branch doc alignment (2026-01-29)

## Scope
Align paper + slides with VINv3 voxel-branch implementation in
`oracle_rri/oracle_rri/vin/model_v3.py`.

## Changes applied
- **Macros:** added symbols for voxel-field derived channels used in v3:
  `counts_norm`, `cent_pr_nms`, `observed`, `unknown`, `new_surface_prior`.
  (`docs/typst/shared/macros.typ`)
- **Slides:** updated EVL/VIN branch slides to reflect:
  - `cent_pr_nms` (not `cent_pr`) used in v3 field bundle,
  - `counts_norm` (not raw counts) in `field_v`,
  - `pts_world` required (not optional),
  - free-space fallback `observed * (1 - occ_input)`.
  (`docs/typst/slides/slides_4.typ`)
- **Paper:** updated architecture section to:
  - list the actual default channel set for v3,
  - specify `cent_pr_nms` and `counts_norm`,
  - document the free-space fallback,
  - note adaptive pooling + center-crop for voxel centers,
  - note uniform weights for voxel-projection FiLM stats.
  (`docs/typst/paper/sections/06-architecture.typ`)

## Key inconsistencies fixed
- Docs referenced `cent_pr` and raw `counts`; v3 uses `cent_pr_nms` and `counts_norm`.
- `pts_world` described as optional; v3 requires it for pos_grid and voxel_proj.
- Free-space fallback not documented previously.

## Follow-ups
- If we ever switch v3 field channels or pooling strategy, update the “default channel set”
  sentence in Section 06.
- Consider adding a short note in the appendix about `pts_world` requirement for cached backbones.

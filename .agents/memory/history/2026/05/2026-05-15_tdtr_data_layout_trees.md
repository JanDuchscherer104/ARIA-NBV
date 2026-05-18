---
id: 2026-05-15_tdtr_data_layout_trees
date: 2026-05-15
title: "tdtr Data-Layout Tree Figures"
status: done
topics: [data-handling, docs, typst, tdtr, rollouts]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/shared/data-layout-trees.typ
  - docs/figures/diagrams/data_handling/typst/
  - docs/figures/diagrams/data_handling/README.md
  - aria_nbv/aria_nbv/data_handling/README.md
---

## Task

Added standalone Typst/tdtr tree figures for ARIA-NBV offline and rollout
store layouts so exact hierarchical schemas can be included in Markdown and
Typst documents.

## Method

Created a shared `docs/typst/shared/data-layout-trees.typ` helper with stable
schema-authored tree functions and rendered each standalone source under
`docs/figures/diagrams/data_handling/typst/` to tracked SVG.

## Outputs

- Added tdtr trees for `vin_offline/`, implemented `rollouts.zarr/`, one
  joined multi-step rollout sample, and the target sharded `rollouts_v1/`
  architecture.
- Revised the implemented `rollouts.zarr/` figure into one connected
  left-to-right tree instead of disconnected table panels.
- Added a top-level offline-to-rollout persisted relation tree and rewired the
  joined rollout sample view into a single connected root.
- Updated the data-handling README to include the new SVG tree renders near
  the matching Markdown tree schemas.
- Kept Mermaid diagrams as relationship/flow views; the tdtr figures own
  exact hierarchical tree layouts.

## Verification

- Compiled every standalone tree source to SVG with `typst compile --root .`
  from `docs/`.
- Compiled a smoke PDF from stdin that imports the shared helper directly.
- Rendered PNG previews to `/tmp` and inspected the rollout, sample, VIN, and
  sharded-tree layouts for bracket leakage, clipping, and readability.

## Canonical State Impact

No canonical memory state update is needed. This adds reusable documentation
figures for the existing data-layout direction.

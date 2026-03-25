---
id: 2026-01-30_vin_architecture_paper_slides_sync_2026-01-30
date: 2026-01-30
title: "Vin Architecture Paper Slides Sync 2026 01 30"
status: legacy-imported
topics: [architecture, sync, 2026, 01, 30]
source_legacy_path: ".codex/vin_architecture_paper_slides_sync_2026-01-30.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VIN scoring architecture: paper ↔ slides sync (2026-01-30)

## Goal
Tighten `docs/typst/paper/sections/06-architecture.typ` by removing repeated prose and make the
`slides_4.typ` “VIN Scoring Architecture” section match the paper notation (VINv3).

## Changes (paper)
- `docs/typst/paper/sections/06-architecture.typ`
  - Compressed “what are components?” repetition: replaced the second overview sentence with a shorter
    “we denote core features …” statement.
  - Compressed 6D-rotation motivation: removed the extra stability sentence (still cited via @zhou2019continuity).
  - Global context attention equations:
    - Kept `#symb.vin.query/#symb.vin.key/#symb.vin.value`.
    - Added explicit origin subscripts by using `(#symb.vin.key)_j^("vox")` and `(#symb.vin.value)_j^("vox")`.
  - Cleaned a small typo (“SSglobal” → “global”).

## Changes (slides)
- `docs/typst/slides/slides_4.typ` (VIN Scoring Architecture)
  - Scene branch: FieldBundle slide now explicitly mentions derived stats (#eqs.vin.counts_norm and #eqs.vin.new_surface_prior).
  - Global context + FiLM slide:
    - Uses `K_j^("vox")` / `V_j^("vox")` to match paper and clarify feature origin.
    - Adds missing FiLM equation via `#eqs.features.film`.
    - Mentions voxel-projection stats as projections of pooled voxel centers into candidate cameras.

## Render verification
- `make typst-paper` and `make typst-slide` succeed.
- Re-rendered and inspected pages (PNG extracted from `docs/typst/slides/slides_4.pdf`):
  - `/tmp/vin_arch_scene_after3/slide-29.png` (FieldBundle with derived stats)
  - `/tmp/vin_arch_scene_after3/slide-31.png` (Global context + FiLM with K/V superscripts and FiLM eq)

## Follow-ups (optional)
- Consider defining dedicated symbols for voxel-projection stats in `macros.typ` (e.g., `bold(s)_"vox"`)
  and using them consistently in both paper and slides (currently described in prose/diagram only).

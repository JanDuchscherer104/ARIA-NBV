---
id: 2026-03-25_vin_nbv_qmd_v3_rewrite
date: 2026-03-25
title: "Rewrite vin_nbv.qmd as the current VINv3 architecture page"
status: done
topics: [docs, quarto, vin, architecture]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/impl/vin_nbv.qmd
artifacts:
  - docs/contents/impl/vin_nbv.html
---

## Task

Replace the legacy `vin_nbv.qmd` page with a concise VINv3-focused architecture
document aligned with the Typst paper, `slides_4.typ`, and
`oracle_rri/oracle_rri/vin/model_v3.py`.

## Method

Read the paper architecture/training sections, the slide deck, the current
`VinModelV3` implementation, its typed outputs, and the best-run W&B artifacts.
Rebuilt the page around the current implementation surface and reused the same
shared architecture figures already used by the paper and slides.

## Findings

- The old page was mostly a VIN history/reference dump and no longer matched the
  implemented model surface.
- The current best run is `rtjvfyyp` / `v03-best`, and its effective VIN config
  differs from class defaults in meaningful ways, especially
  `use_traj_encoder=true` and a reduced `scene_field_channels` selection.
- The slide figures already exist as shared assets under `docs/figures`, so no
  new diagram generation was needed.

## Verification

- `quarto render docs/contents/impl/vin_nbv.qmd --to html`
- Confirmed rendered HTML contains all reused figure paths.
- Confirmed the rewritten page no longer references legacy surfaces such as
  `experimental/model.py`, `model_v2.py`, shell-descriptor notes, or old VIN
  version labels.
- `quarto check` hit an unrelated environment warning due a missing external
  Jupyter kernel path (`/home/jandu/repos/traenslenzor/.venv/bin/python3`), not
  a page-specific render failure.

## Canonical State Impact

No canonical state files changed. This was a documentation-alignment update for
an existing model surface.

## Prompt Follow-Through

This note predates the privileged owner-directive memory contract. No additional durable owner prompt items were backfilled here beyond any canonical state updates already recorded in this debrief.

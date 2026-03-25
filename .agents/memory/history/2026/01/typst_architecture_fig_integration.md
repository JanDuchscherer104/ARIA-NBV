---
id: 2026-01-30_2026-01-30_typst_architecture_fig_integration
date: 2026-01-30
title: "Typst Architecture Fig Integration"
status: legacy-imported
topics: [typst, architecture, fig, integration]
source_legacy_path: ".codex/2026-01-30_typst_architecture_fig_integration.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# 2026-01-30 — Typst paper: architecture section figure integration

## Task
- Inspect rendered pages for `docs/typst/paper/sections/06-architecture.typ`.
- Inspect + interpret all diagrams/figures used there.
- Improve figure↔text integration (avoid float gaps; avoid tiny/compressed diagrams).
- Ensure the architecture narrative matches the conceptual branch flow in `oracle_rri/oracle_rri/vin/model_v3.py`.
- Prefer shared notation via `docs/typst/shared/macros.typ` (`#symb`, `#eqs`).

## What I checked
- Generated fresh repo snapshot: `make context` → `.codex/codex_make_context.md`.
- Built artifacts:
  - `make typst-paper` → `docs/typst/paper/main.pdf`
  - `make typst-slide` (defaults to `slides_4.typ`) → `docs/typst/slides/slides_4.pdf`
- Render inspection (text-based): searched `main.pdf` page-by-page with `pdftotext` for section title and figure captions to confirm figures stay near the relevant subsections.

## Key findings (render)
- `= Aria-VIN-NBV Architecture` starts on page 9 of `docs/typst/paper/main.pdf`.
- Architecture figures now appear close to their subsections (no long-distance float drift):
  - Inputs + EVL summaries: page 10
  - Pose encoding: page 11
  - Global pooling + semidense stats: page 12
  - Grid CNN + trajectory context: page 14
  - Head/CORAL: page 15

## Changes made
- `docs/typst/paper/sections/06-architecture.typ`
  - Removed the compressed multi-diagram grid that caused unreadable diagrams and severe float drift.
  - Placed each branch diagram next to its corresponding subsection and forced local placement (`placement: none`) to avoid IEEE float gaps.
  - Added short “see @fig:…” bridge sentences so each figure is explicitly referenced by the surrounding text.
  - Replaced repeated equations with shared `#eqs` entries (counts normalization, new-surface prior, FiLM, candidate validity mask, semidense validity/visibility).
  - Clarified the CORAL head outputs: expected ordinal score `#symb.vin.rri_hat` is used as ranking proxy; continuous expected RRI uses bin values (delegated to *Training Objective* section).
- `docs/typst/shared/macros.typ`
  - Added `#eqs.vin.counts_norm` and `#eqs.vin.new_surface_prior`.
  - Updated `#eqs.features.film` to use `#symb.vin.*` symbols (notation consistency).
- `docs/figures/diagrams/vin_nbv/mermaid/head_paper.{mmd,png}`
  - Added a paper-friendly (column-friendly) head/CORAL diagram to replace the very wide original head diagram.
- `docs/typst/paper/sections/10a-entity-aware.typ`
  - Fixed a duplicate Typst label (`<sec:entity-aware>`) that prevented paper compilation.
- `.codex/AGENTS_INTERNAL_DB.md`
  - Documented the “expected class vs expected RRI” gotcha: `VinPrediction.expected` is an expected **class** score; continuous expected RRI uses bin reps `u_k`.

## Follow-ups / suggestions
- Consider replacing/moving the EVL “rich tree” screenshot (`@fig:evl-summary`) to an appendix or converting it to a small table of named tensor channels; it is useful but visually heavy.
- Consider sourcing “Current v3 baseline” hyperparameters from a single exported config artifact (to avoid drift between paper and `slides_4.typ`).

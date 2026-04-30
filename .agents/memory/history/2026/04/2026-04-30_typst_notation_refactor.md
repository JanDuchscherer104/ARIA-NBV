---
id: 2026-04-30_typst_notation_refactor
date: 2026-04-30
title: "Typst notation refactor and Quarto notation shortcodes"
status: done
topics: [docs, typst, quarto, notation]
confidence: high
canonical_updates_needed: []
---

## Task

Restore the seminar paper compile after the Ubuntu migration, split the shared Typst macro surface into maintainable modules, and expose shared symbols/equations to Quarto pages through shortcodes.

## Method

Fixed paper sections that still loaded the legacy `offline_cache_stats.json` wrapper to use the current `vin_offline_store_stats.json` artifact. Split `docs/typst/shared/macros.typ` into a compatibility facade over focused style, term, symbol, math, and equation modules, with domain-level symbol/equation files. Added `docs/notation.yml` plus generated Lua/Typst notation artifacts through `scripts/glossary_build.py`, and extended the existing `aria-glossary` shortcode extension with `sym` and `eq`.

## Outputs

Existing Typst imports through `../shared/macros.typ` remain the stable public surface. Quarto pages can now use `{{< sym oracle.rri >}}` for inline notation and `{{< eq rri.rri >}}` for display equations.

## Verification

- `make glossary`
- `cd docs && typst compile typst/paper/main.typ /tmp/aria-nbv-paper.pdf --root .`
- `cd docs && typst compile typst/slides/slides_4.typ /tmp/aria-nbv-slides-4.pdf --root .`
- `cd docs && typst compile typst/slides/slides_thesis_outlook.typ /tmp/aria-nbv-slides-thesis-outlook.pdf --root .`
- Temporary Quarto smoke page under `docs/` rendered with `{{< sym oracle.rri >}}` and `{{< eq rri.rri >}}`.

The slide compiles still warn that `Open Sans` is unavailable in the local font environment; this is pre-existing and non-fatal.

## Canonical State Impact

No canonical state update is needed. This was a docs/tooling maintainability change that preserves the current VIN offline-store and paper narrative contracts.

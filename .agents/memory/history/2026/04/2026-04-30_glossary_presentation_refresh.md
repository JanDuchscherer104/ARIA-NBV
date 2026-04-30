---
id: 2026-04-30_glossary_presentation_refresh
date: 2026-04-30
title: "Glossary Presentation Refresh"
status: done
topics: [docs, glossary, quarto, kg, typst]
confidence: high
canonical_updates_needed: []
files_touched:
  - scripts/glossary_build.py
  - docs/contents/glossary.qmd
  - docs/_quarto.yml
  - docs/styles.css
---

## Task

Refresh the shared glossary presentation from a long numbered outline into a reader-facing reference page while preserving `docs/glossary/terms.yml` as the only hand-authored glossary source and keeping Typst/KG outputs generated from the same records.

## Method

`scripts/glossary_build.py` now writes `docs/contents/glossary.qmd` directly from the YAML source instead of depending on a generated include. The generated page uses category chips, compact term cards, stable anchors on card containers, reader-friendly related/doc labels, and collapsible metadata for related terms, docs, and references.

`docs/styles.css` is hand-authored, scoped to `.glossary-*`, and included after the generated quartodoc stylesheet in `docs/_quarto.yml`. The mobile rules keep chips wrapped and cap the card width for the Chrome headless 390px screenshot path, whose layout viewport bottoms out at 500px.

## Verification

Passed `make glossary`, Quarto renders for `contents/glossary.qmd` and `contents/questions.qmd`, `typst compile typst/paper/main.typ --root .`, headless desktop/mobile screenshots, `scripts/nbv_qmd_outline.sh --compact`, `make kg-materialize`, `make check-agent-memory`, and `git diff --check`.

## Canonical State Impact

No canonical memory update is needed beyond the existing shared-glossary source record; this pass changes presentation and generation mechanics, not the project glossary source-of-truth policy.

---
id: 2026-05-19_sampling_target_selection_theory_docs
date: 2026-05-19
title: "Sampling And Target-Selection Theory Docs"
status: done
topics: [docs, pose-generation, target-selection, mermaid, docstrings]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/theory/candidate_sampling_target_selection.qmd
  - docs/_quarto.yml
  - docs/contents/diagrams.qmd
  - docs/contents/theory/rl_planning.qmd
  - docs/contents/thesis/questions.qmd
  - docs/contents/thesis/roadmap.qmd
  - docs/figures/diagrams/pose_generation/
  - aria_nbv/aria_nbv/data_handling/_target_selection.py
  - aria_nbv/aria_nbv/pose_generation/
---

## Task

Implemented the public theory documentation plan for target-conditioned sampling
and target selection. The new Quarto page explains actor-visible target
selection, candidate center sampling, orientation/pruning, mixture components,
and rollout branch selection with the main formulas used by the current thesis
contract.

## Method

Added Mermaid source diagrams and rendered SVGs under
`docs/figures/diagrams/pose_generation/mermaid/`, linked the new theory page
from the Theory sidebar, RQ page, roadmap, rollout theory page, and diagram
index, and enriched the priority Python module docstrings with compact theory
contracts.

## Verification

Ran targeted `ruff format` and `ruff check` on the changed Python surfaces.
Linted all new Mermaid sources with `tools/mermaid/scripts/aria_mermaid_lint.py`.
Rendered the new Quarto page, diagram index, rollout theory page, research
questions, and roadmap. `mmdc` was not installed, so SVGs were rendered through
`npx @mermaid-js/mermaid-cli` with a temporary Puppeteer config and the local
headless shell.

Follow-up contrast fix on 2026-05-19: Mermaid `htmlLabels` applied semantic
`classDef fill` values to label spans, which made text too pale against pastel
node fills. Added a higher-specificity `themeCSS` text rule for node labels,
kept the canonical semantic `classDef` definitions unchanged, rerendered SVGs,
and re-ran Mermaid lint plus Quarto renders for the affected pages.

## Canonical State Impact

No canonical state update is required. The public docs now own the full
sampling/target-selection explanation; generated API docs inherit the compact
docstring contracts.

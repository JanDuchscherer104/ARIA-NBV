---
id: 2026-05-15_data_handling_readme_diagram_package
date: 2026-05-15
title: "Data-Handling README Diagram Package"
status: done
topics: [data-handling, rollouts, docs, mermaid]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/data_handling/README.md
  - docs/figures/diagrams/data_handling/
---

## Task

Revised the data-handling README into a clearer developer-facing reference for
the immutable VIN offline store and the target-conditioned rollout replay
store.

## Method

Moved README Mermaid content into a dedicated diagram package under
`docs/figures/diagrams/data_handling/`, with `.mmd` sources and tracked SVG
renders. The README now imports SVGs and keeps exact filesystem/table layouts
as Markdown trees.

## Outputs

- Added diagrams for data-store architecture, physical layouts, rollout table
  indexing, a joined multi-step sample tree, and rollout-generation sequence.
- Documented branch/time ownership explicitly: target top-k, policy recipes,
  retained `chain_id`, `step_index`, and candidate `shell_index`.
- Kept the current standalone `rollouts.zarr` implementation distinct from the
  target sharded `rollouts_v1/` architecture.
- Re-rendered the diagrams after normalizing Mermaid/KaTeX labels to the
  existing escaped-backslash convention (`\\textbf`, `\\texttt`, `\\\\\\texttt`
  after line breaks), so the SVG output no longer exposes raw LaTeX commands.

## Verification

- `python3 tools/mermaid/scripts/aria_mermaid_lint.py docs/figures/diagrams/data_handling/mermaid/*.mmd`
- Rendered SVGs with `npx -y @mermaid-js/mermaid-cli` using the system Chrome
  executable through a temporary Puppeteer config.
- Checked that the README contains no inline Mermaid fences and that all SVG
  links resolve.

## Canonical State Impact

No canonical memory state update is needed. This is documentation packaging and
clarification of the existing data-layout direction.

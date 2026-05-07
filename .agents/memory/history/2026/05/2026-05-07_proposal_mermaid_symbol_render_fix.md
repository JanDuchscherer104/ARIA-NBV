---
id: 2026-05-07_proposal_mermaid_symbol_render_fix
date: 2026-05-07
title: "Proposal Mermaid Symbol Render Fix"
status: done
topics: [docs, typst, proposal, mermaid]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/figures/proposal_system_flow.mmd
  - docs/typst/thesis/figures/proposal_system_flow.png
  - docs/typst/thesis/figures/proposal_gantt.mmd
  - docs/typst/thesis/figures/proposal_gantt.png
  - docs/typst/thesis/proposal.pdf
---

## Task

Ensure TeX-like symbols in proposal Mermaid figures render as symbols instead
of literal underscore/caret text.

## Method

Changed the system-flow Mermaid diagram to use HTML labels with rendered
subscripts, superscripts, Greek letters, and bold variable symbols. The Mermaid
Gantt renderer prints HTML tags literally, so the raw `Q_H` task label was
renamed to avoid unrendered TeX syntax there.

## Verification

- `PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome npx -y @mermaid-js/mermaid-cli -i docs/typst/thesis/figures/proposal_system_flow.mmd -o docs/typst/thesis/figures/proposal_system_flow.png -s 4`
- `PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome npx -y @mermaid-js/mermaid-cli -i docs/typst/thesis/figures/proposal_gantt.mmd -o docs/typst/thesis/figures/proposal_gantt.png -s 4`
- `make proposal-pdf`
- Visual PDF inspection of Figure 1 and Figure 2 pages.

## Canonical State Impact

No canonical state update needed. This was a proposal figure-rendering fix.

---
id: 2026-05-07_proposal_flowchart_layout_fix
date: 2026-05-07
title: "Proposal Flowchart Layout Fix"
status: done
topics: [docs, typst, proposal, mermaid]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/sections/proposal/02-related-work.typ
  - docs/typst/thesis/figures/proposal_system_flow.mmd
  - docs/typst/thesis/figures/proposal_system_flow.png
  - docs/typst/thesis/proposal.pdf
---

## Task

Fix the proposal related-work lineage and Figure 1 layout after the rendered PDF
showed an overlong prose chain and an oversized Mermaid flowchart.

## Method

Replaced the prose lineage with a compact Typst math chain
`cal(U)_"cov/unc" -> hat(r)_t^e (i) -> r_t^e -> G_t^((H)) -> Q_(H,theta)`.
Reworked the Mermaid figure into a left-to-right symbolic evidence chain with
compact node labels and a dashed bridge node, then reduced the included figure
width in Typst.

## Verification

- `PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome npx -y @mermaid-js/mermaid-cli -i docs/typst/thesis/figures/proposal_system_flow.mmd -o docs/typst/thesis/figures/proposal_system_flow.png -s 4`
- `cd docs && typst compile typst/thesis/proposal.typ --root . /tmp/proposal-figure-check.pdf`
- Visual page inspection via `pdftoppm`/image view for the lineage, Figure 1,
  and the temperature-softmax denominator.
- `make proposal-pdf`

## Canonical State Impact

No canonical project-state update needed. This was a notation/layout repair for
the advisor-facing proposal.

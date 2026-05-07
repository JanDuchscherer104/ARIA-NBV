---
id: 2026-05-07_proposal_qh_bold_math_fix
date: 2026-05-07
title: "Proposal QH Bold Math Fix"
status: done
topics: [docs, typst, proposal, notation]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/sections/proposal/02-problem.typ
  - docs/typst/thesis/sections/proposal/03-objectives.typ
  - docs/typst/thesis/sections/proposal/04-method.typ
  - docs/typst/thesis/proposal.pdf
---

## Task

Fix proposal notation so vector, matrix, and tensor symbols render in bold, and
verify the masked Double-Q target equation for `y_t` renders correctly.

## Method

Replaced fragile target-network notation with `theta^-`, used Typst `bold(...)`
for state, point evidence, target descriptors, candidate tables, mask/reason
vectors, token tensors, embeddings, and projection weights, and separated the
scalar softmax logit `ell_(t,i)` from vector embedding `bold(u)_(t,i)`.

## Verification

- `cd docs && typst compile typst/thesis/proposal.typ --root . /tmp/proposal-bold-final.pdf`
- Visual PDF page check via `pdftoppm`/image inspection for the `Q_H` and `y_t`
  equations.
- `make proposal-pdf`

## Canonical State Impact

No canonical project-state update needed. This was a proposal notation and
rendering correction, not a thesis-scope change.

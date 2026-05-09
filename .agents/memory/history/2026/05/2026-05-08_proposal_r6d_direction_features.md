---
id: 2026-05-08_proposal_r6d_direction_features
date: 2026-05-08
title: "Proposal R6D and Directional Visibility Features"
status: done
topics: [thesis, typst, proposal, notation, q_h]
confidence: medium
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/sections/proposal/_style.typ
  - docs/typst/thesis/sections/proposal/04-method.typ
  - docs/typst/shared/symbols/vin.typ
  - docs/typst/shared/equations/features.typ
---

## Task

Folded the R6D versus visibility-direction design into the advisor-facing
proposal. The proposal now treats the continuous 6D rotation representation as
a candidate pose/orientation feature and keeps accumulated visibility as an
actor-visible directional memory on `S^2`, represented by low-order spherical
harmonic coefficients or a second-moment summary.

## Method

Added shared Typst symbols and equations for directional unit vectors,
spherical-harmonic directional memory, moment directional memory, and candidate
directional novelty. The method section now states that summing R6D pose codes
is not the right visibility-memory model; R6D belongs in the pose branch, while
directional observation history belongs in the visibility branch.

Enabled proposal-wide equation numbering for the current elegance-over-compactness
preference. The wide replay tuple collided with its equation number, so it was
rewritten as a named replay record with explicit identifier and metadata fields.

## Verification

- `cd docs && typst compile typst/thesis/proposal.typ /tmp/proposal-r6d-direction-numbered.pdf --root .`
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/sections/proposal/04-method.typ docs/typst/thesis/sections/proposal/_style.typ`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/proposal.typ -o /tmp/proposal-numbered-pages-method --root docs --pages 11-14 --ppi 220`
- `git diff --check -- docs/typst/thesis/sections/proposal/04-method.typ docs/typst/thesis/sections/proposal/_style.typ docs/typst/shared/equations/features.typ docs/typst/shared/symbols/vin.typ`
- `make kg-claim-check KG_CLAIM='The proposal treats 6D rotation representation as a planned candidate-pose feature for Q_H, while accumulated visibility direction is represented separately as actor-visible S^2 directional memory or moment features; this is a planned feature branch, not an implemented result.'`

The KG claim check returned `unverifiable` because canonical sources currently
support directional observability and Q_H feature-ablation planning in general,
but do not yet canonically lock the exact R6D/S2 design. The proposal text keeps
the branch planned rather than implemented.

## Canonical State Impact

No immediate canonical state edit was made. The proposal now contains a sharper
planned architecture detail that should be promoted only if it survives advisor
review.

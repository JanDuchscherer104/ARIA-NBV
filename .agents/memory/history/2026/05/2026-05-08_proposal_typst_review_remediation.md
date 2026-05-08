---
id: 2026-05-08_proposal_typst_review_remediation
date: 2026-05-08
title: "Proposal Typst Review Remediation"
status: done
topics: [proposal, typst, notation, mermaid, thesis]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/shared/symbols/rl.typ
  - docs/typst/shared/symbols/entity.typ
  - docs/typst/shared/symbols/vin.typ
  - docs/typst/shared/symbols/obs.typ
  - docs/typst/shared/equations/entity.typ
  - docs/typst/shared/equations/rl.typ
  - docs/typst/thesis/sections/proposal/02-problem.typ
  - docs/typst/thesis/sections/proposal/02-related-work.typ
  - docs/typst/thesis/sections/proposal/03-objectives.typ
  - docs/typst/thesis/sections/proposal/04-method.typ
  - docs/typst/thesis/figures/proposal_system_flow.mmd
  - docs/typst/thesis/figures/proposal_system_flow.png
---

## Task

Remediated the advisor proposal Typst review findings from 2026-05-08: shared
notation drift, Transformer output indexing, endpoint metric consistency, and
the compact system-flow figure.

## Method

Moved recurring proposal notation into `docs/typst/shared` facades, including
actor/oracle/counterfactual states, target descriptor/error/reward/return/gain,
lookahead headroom, recovery fraction, and candidate-query `Q_H` equations.
Proposal sections now call shared symbols/equations instead of repeating fragile
inline math. The Mermaid source was migrated to the ARIA semantic class palette
and the PNG was regenerated locally.

## Findings

The main Typst attachment issue was fixed by grouping Transformer output before
candidate indexing. The broader Project Aria/ASE/EFM3D claim was too broad for
the KG checker as originally phrased, so the proposal wording was narrowed to
the supported actor-visible historic-state and GT-exclusion contract.

## Verification

- `cd docs && typst compile typst/thesis/proposal.typ /tmp/proposal-fixed.pdf --root .`
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/sections/proposal`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/proposal.typ -o /tmp/proposal-fixed-pages --root docs --pages 5-11 --ppi 300`
- `aria_nbv/.venv/bin/python tools/mermaid/scripts/aria_mermaid_lint.py docs/typst/thesis/figures/proposal_system_flow.mmd`
- `npx -y @mermaid-js/mermaid-cli -i docs/typst/thesis/figures/proposal_system_flow.mmd -o docs/typst/thesis/figures/proposal_system_flow.png -b white -w 1600 -p /tmp/aria-nbv-mmdc-puppeteer.json`
- `make kg-claim-check` returned supported for the VIN-NBV precedent claim, the
  actor-visible historic-state / GT-exclusion claim, and the planned `Q_H`
  thesis-core claim.

## Canonical State Impact

No canonical state update is needed. The thesis direction and scope remain
unchanged; this pass only repaired proposal notation, rendering, and grounding.

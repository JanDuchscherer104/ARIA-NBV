---
id: 2026-05-12_qcnet_rpe_advisor_distillation
date: 2026-05-12
title: "QCNet-Inspired RPE In Advisor Distillation"
status: done
topics: [typst, advisor-distillation, qh, qcnet, notation]
confidence: medium
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
  - docs/typst/shared/equations/features.typ
---

## Task

Patched the advisor-facing distillation with a docs/design-only representation contract for the planned finite-candidate `Q_H`: R6D+LFF candidate pose features, QCNet-inspired candidate-query relative positional encodings, and separate `S^2` accumulated directional memory.

## Method

Added shared Typst equations for candidate-local relative frames, compact relative positional encodings, and edge-conditioned candidate attention. The handout now cites QCNet only as an inspiration for query-centric relative encodings and explicitly excludes QCNet trajectory decoding, motion-forecasting losses, and streaming claims from the thesis core.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-distillation-qcnet-rpe.pdf --root .`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-qcnet-rpe-pages --root docs --pages 5-12 --ppi 220`
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ docs/typst/shared/equations/features.typ`
- `rg -n "TODO|FIXME|\\?\\?\\?|QCNet artifact" docs/typst/thesis/advisor_distillation.typ docs/typst/shared/equations/features.typ`

KG claim checks were attempted for QCNet inspiration scope, R6D/RPE/S2 separation, and planned-versus-implemented status. All returned `unverifiable` because the current literature graph has `paper:*` nodes without source paths, so those checks could not support or contradict the claims.

## Canonical State Impact

No canonical state update is required. This is a handout-level architecture proposal and shared-equation addition, not implemented thesis evidence.

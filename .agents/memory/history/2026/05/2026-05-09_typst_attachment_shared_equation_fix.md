---
id: 2026-05-09_typst_attachment_shared_equation_fix
date: 2026-05-09
title: "Typst Attachment Fixes For Shared Equations"
status: done
topics: [typst, thesis, notation, equations]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/shared/equations/coverage.typ
  - docs/typst/shared/equations/entity.typ
  - docs/typst/shared/equations/features.typ
  - docs/typst/shared/equations/rl.typ
  - docs/typst/shared/equations/rri.typ
  - docs/typst/shared/equations/vin.typ
  - docs/typst/thesis/advisor_distillation.typ
  - docs/typst/thesis/sections/proposal/04-method.typ
---

## Task

Fixed Typst math attachment over-capture where function arguments immediately
followed subscripts or superscripts, including the directional-memory equations
in `docs/typst/shared/equations/features.typ`.

## Method

Inserted explicit spaces after attached labels for subscripted function calls
such as `w_k (v)`, `"Q"_H (...)`, `"RRI"_e (...)`, `"Huber"_1 (...)`, and
`phi_"dir" (...)`. Grouped the directional novelty transpose as
`(bold(d)_(t,i) (bold(v)))^top` so the transpose applies to the full evaluated
direction vector rather than to the argument.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-distillation-attachment-fix.pdf --root .`
- `cd docs && typst compile typst/thesis/proposal.typ /tmp/proposal-attachment-fix.pdf --root .`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-attachment-fix-pages --root docs --pages 6-7 --ppi 220`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/proposal.typ -o /tmp/proposal-attachment-fix-pages --root docs --pages 5-8 --ppi 220`
- Regex scan for subscripted or superscripted calls found no remaining attachment leak candidates except `op("softplus")(...)`, which is not an attached subscript/superscript call.

Strict hygiene still reports `Q_(H,theta)` in `docs/typst/shared/equations/rl.typ`
as "notation that should migrate to shared modules"; this is a false positive
because the match is already inside the shared equation module.

## Canonical State Impact

No canonical state update is needed. This fixes Typst rendering hygiene without
changing the thesis scope or notation semantics.

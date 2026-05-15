---
id: 2026-05-15_advisor_distillation_theory_sharpening
date: 2026-05-15
title: "Advisor Distillation Theory Sharpening"
status: done
topics: [thesis, typst, advisor-handout, literature-review, q-h]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
  - docs/typst/thesis/advisor_distillation.pdf
  - .agents/memory/history/2026/05/2026-05-15_advisor_distillation_theory_sharpening.md
---

## Task

Implemented the focused 2026-05-15 advisor-distillation theory-sharpening pass for `docs/typst/thesis/advisor_distillation.typ`. The requested scope was to preserve the existing six-RQ structure and geometric-ML section while making the memo read as a principled target-RRI and finite-candidate `Q_H` thesis rather than an architecture inventory.

## Method

Tightened the opening thesis contract around four contributions: leakage-safe target-RRI protocol, oracle-headroom finite-candidate planning, geometry-aware residual `Q_H`, and support-aware rollout generation. Kept the evaluation sequence conditional on positive oracle-lookahead headroom and preserved the negative-result path for near-zero headroom.

Added object-centric 3DGS NBV as a narrow target-focus contrast in the adoption ledger using the existing `ObjectCentricNBV-jeong2026` bibliography key. The source role is limited to object-level view utility and separate target reporting; it does not replace ASE mesh-supervised target-RRI, EVL/predicted-OBB target inputs, or the finite-candidate objective.

Kept POp-GS, ActiveGAMER, R3-RECON, SA-ResGS, proxy objectives, full simulators, point backbones, and continuous control outside the thesis core. No public API, schema, Python interface, or bibliography key was changed.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ typst/thesis/advisor_distillation.pdf --root .` passed.
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-theory-sharpen-pages --root docs --pages 1-18 --ppi 220` passed.
- Visual inspection covered the opening claim, geometric-bias table, support-coverage table, literature/adoption ledger, roadmap risks, and bibliography pages.
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ docs/typst/shared` passed blocking checks; remaining output was advisory shared-notation/glossary review prompts.
- `make kg-claim-check KG_CLAIM="ARIA-NBV's advisor-facing thesis core is target-specific RRI with actor-visible target conditioning, oracle-lookahead headroom, and a finite-candidate Q_H value model; continuous control, external simulators, and proxy objectives are lower-priority extensions."` -> supported, confidence 1.0.
- `make kg-claim-check KG_CLAIM="Object-centric and active 3D Gaussian Splatting NBV papers support target/object-focused and uncertainty-view-utility contrasts, but they should not replace ARIA-NBV's ASE mesh-supervised target-RRI objective."` -> supported, confidence 1.0.

## Canonical State Impact

No canonical memory update is needed. The revision restates the locked thesis direction: target-specific RRI, actor-visible target conditioning, oracle-lookahead headroom, finite-candidate `Q_H`, and bridge/future-work boundaries for simulators, active 3DGS proxy objectives, and continuous control.

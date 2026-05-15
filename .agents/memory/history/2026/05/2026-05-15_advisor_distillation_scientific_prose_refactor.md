---
id: 2026-05-15_advisor_distillation_scientific_prose_refactor
date: 2026-05-15
title: "Advisor Distillation Scientific-Prose Refactor"
status: done
topics: [thesis, typst, advisor-handout, scientific-writing, q-h]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
  - docs/typst/thesis/advisor_distillation.pdf
  - .agents/memory/history/2026/05/2026-05-15_advisor_distillation_scientific_prose_refactor.md
---

## Task

Implemented the structured scientific-prose refactor requested on 2026-05-15 for `docs/typst/thesis/advisor_distillation.typ`. The goal was to keep the six-RQ structure and thesis direction intact while making the advisor handout read as a scientific research contract: testable claim, method, evidence gate, and limitation.

## Method

Tightened the opening around a single target-conditioned planning hypothesis and replaced managerial phrasing with scope, evidence, and limitation language. Split the formal-model section into state/visibility, target-specific RRI labels, candidate transitions, and research-question/headroom subsections without changing the equations.

Reframed the value-model section as a finite-candidate value-model hypothesis with controls and ablations. Standardized prose terminology around target-specific RRI while keeping shorter target RRI wording only in compact table cells. Compressed the literature ledger introduction so source roles remain narrow, including object-centric 3DGS as a target-focus contrast rather than a replacement objective. Reworded the risk section so near-zero headroom diagnoses target matching, candidate support, and supervision scale before added model complexity.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ typst/thesis/advisor_distillation.pdf --root .` passed.
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-prose-refactor-pages --root docs --pages 1-20 --ppi 220` passed.
- Visual inspection covered the opening contract, formal-model subsections, value-model section, adoption ledger, and roadmap/risk page.
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ docs/typst/shared` passed blocking checks; remaining output was advisory shared-glossary/shared-notation review prompts.
- `make kg-claim-check KG_CLAIM="ARIA-NBV's advisor-facing thesis core is target-specific RRI with actor-visible target conditioning, oracle-lookahead headroom, and a finite-candidate Q_H value model; continuous control, external simulators, and proxy objectives are lower-priority extensions."` -> supported, confidence 1.0.
- `make kg-claim-check KG_CLAIM="Object-centric and active 3D Gaussian Splatting NBV papers support target/object-focused and uncertainty-view-utility contrasts, but they should not replace ARIA-NBV's ASE mesh-supervised target-RRI objective."` -> supported, confidence 1.0.
- `git diff --check -- docs/typst/thesis/advisor_distillation.typ` passed.

## Canonical State Impact

No canonical memory update is needed. The edit restates the locked thesis direction and does not change public APIs, schemas, bibliography keys, glossary entries, or Python interfaces.

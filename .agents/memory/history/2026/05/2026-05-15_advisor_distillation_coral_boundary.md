---
id: 2026-05-15_advisor_distillation_coral_boundary
date: 2026-05-15
title: "Advisor Distillation CORAL Boundary"
status: done
topics: [docs, typst, thesis, coral]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
  - docs/typst/thesis/advisor_distillation.pdf
---

## Task

Clarified the CORAL paragraph in the advisor distillation handout so it cites the original CORAL paper and adopted `coral-pytorch` implementation, then states ARIA-NBV's RRI-specific modifications without turning them into a thesis objective.

## Method

Replaced the prior vague CORAL-interface paragraph with concise advisor-facing prose that separates literature support, implementation provenance, and ARIA-NBV calibration deltas. The text now describes empirical quantile bins, CORAL threshold targets, cumulative-probability decoding, class-marginal scalar recovery, monotone learnable bin representatives, prior-based bias initialization, optional balanced/focal threshold losses, and monotonicity plus relative-to-random diagnostics.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ typst/thesis/advisor_distillation.pdf --root .`
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ docs/typst/shared`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-coral-pages --root docs --pages 8-10 --ppi 220`
- Visual QA inspected `/tmp/advisor-coral-pages/09.png` and `/tmp/advisor-coral-pages/10.png`; the paragraph, equation, and continuation render cleanly.
- `make kg-claim-check KG_CLAIM="ARIA-NBV's CORAL implementation builds on CORAL/coral-pytorch and extends it for RRI with empirical quantile ordinal bins, cumulative-threshold to class-marginal decoding, expected RRI from bin representatives, monotone learnable bin values, prior-based threshold bias initialization, optional balanced/focal threshold losses, and monotonicity plus relative-to-random diagnostics."` returned `unverifiable (confidence=0.2)` with no supporting or contradicting sources because literature `paper:*` nodes lack source paths today.

## Canonical State Impact

No canonical thesis-state update is needed. The change is a local advisor-handout wording/citation clarification grounded in existing bibliography keys and implementation evidence.

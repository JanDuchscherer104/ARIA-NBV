---
id: 2026-05-09_typst_shared_notation_convention
date: 2026-05-09
title: "Typst Shared Notation Convention"
status: done
topics: [typst, thesis, notation, proposal]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - docs/typst/shared/
  - docs/typst/thesis/advisor_distillation.typ
  - docs/typst/thesis/sections/proposal/
  - docs/notation.yml
  - docs/typst/shared/glossary.typ
  - .agents/skills/typst-authoring/
---

## Task

Implemented the agreed ARIA-NBV Typst notation convention after the
symbol/equation review: abstract geometry is calligraphic, implementation
tensors are bold, candidate sets are separated from value functions, abstract
states are non-bold, and thesis-core RRI uses point-mesh error `D` rather than
generic `CD`.

## Method

Updated shared symbols/equations first, then migrated advisor-facing Typst
surfaces and regenerated glossary/notation artifacts with `make glossary`.
Integrated the convention into the repo-local `typst-authoring` skill,
including migration guidance and strict hygiene checks.

## Verification

Compiled the advisor handout and proposal after the shared notation edits:
`cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-test.pdf --root .`
and
`cd docs && typst compile typst/thesis/proposal.typ /tmp/proposal-test.pdf --root .`.
Additional final hygiene and memory checks were run in the same implementation
pass.

## Canonical State Impact

`.agents/memory/state/DECISIONS.md` now records the locked notation convention
so future proposal/thesis edits and the `typst-authoring` skill use the same
source of truth.

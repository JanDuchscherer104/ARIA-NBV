---
id: 2026-05-11_decisions_litkg_cleanup
date: 2026-05-11
title: "Decisions LitKG Cleanup"
status: done
topics: [decisions, docs, litkg, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/references/human_owner_intent.md
  - docs/contents/thesis/roadmap.qmd
  - docs/contents/thesis/questions.qmd
  - docs/contents/literature/vin_nbv.qmd
  - docs/contents/literature/project_aria.qmd
  - docs/contents/literature/efm3d.qmd
  - docs/contents/literature/pb_nbv.qmd
  - docs/contents/theory/surface_metrics.qmd
---

## Task

Apply the LitKG-backed `DECISIONS.md` audit by correcting stale generated
context, retained-QMD, and retired implementation-page references.

## Method

Used LitKG searches and claim checks to compare canonical decisions against
source-order owners, current thesis pages, generated-docs configuration, and
tracked/ignored artifact status. Preserved unrelated dirty worktree changes.

## Findings

The thesis, target-RRI, rollout/Q, VIN/offline, notation, inspection, LRZ, and
proposal/CI decisions remained supported. The stale pressure was limited to
generated-context wording that ignored the tracked `glossary.jsonl` exception
and public docs wording/links that still treated retired manual implementation
QMD pages as active public docs. Public thesis, theory, and literature links
that still pointed at retired `../impl/*.qmd` pages were updated to current
theory pages, generated API pages, or the M1 contract report.

## Verification

Passed `make check-agent-memory`, `make qmd-frontmatter-check`, scoped
`git diff --check`, and targeted Quarto renders for `roadmap.qmd`,
`questions.qmd`, `surface_metrics.qmd`, `vin_nbv.qmd`, `project_aria.qmd`,
`efm3d.qmd`, and `pb_nbv.qmd`. The first roadmap render exposed a stale
`VinDataModule` API link that was already corrected to `reference/lightning.qmd`
in the working tree before the final roadmap render passed without that warning.
KG searches for generated routing context and retired implementation contracts
now return canonical `DECISIONS.md` entries first.

## Canonical State Impact

Canonical memory now distinguishes untracked generated routing context from
tracked glossary/KG artifacts and records that generated API implementation
contracts replace retired manual implementation pages in public docs.

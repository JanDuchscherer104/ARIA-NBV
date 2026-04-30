---
name: docs-curator
description: Use when ARIA-NBV work changes README, SETUP, Quarto docs, Typst paper/slides, bibliography, generated context, public navigation, or canonical memory alignment.
---

# Docs Curator

## When To Use

Use this skill for documentation or narrative work that crosses any of:

- `README.md`, setup docs, Quarto pages, Typst paper, or slides
- bibliography and literature references
- generated context routing artifacts
- canonical state docs or debriefs
- public/internal docs boundary decisions

Do not use it for a localized typo fix unless source-of-truth alignment is at risk.

## Read First

1. `docs/AGENTS.md`
2. `docs/typst/paper/main.typ`
3. `.agents/memory/state/PROJECT_STATE.md`
4. `.agents/memory/state/GOTCHAS.md` when behavior or workflow claims are involved
5. `.agents/references/agent_memory_templates.md` for debriefs

## Rules

- Treat the Typst paper as the highest-level project narrative.
- Keep public Quarto docs reader-facing; internal agent guidance, generated context, and OMX runtime notes stay under `.agents/` unless explicitly curated.
- Link to canonical state or owning implementation docs instead of repeating long explanations.
- Keep bibliography additions in `docs/references.bib`.
- Update canonical memory when the current truth changes; otherwise record `canonical_updates_needed: []` in the debrief.

## Verification

- `cd docs && quarto render contents/roadmap.qmd contents/questions.qmd` for roadmap/question edits
- `cd docs && typst compile typst/paper/main.typ --root .` for paper edits
- `cd docs && typst compile typst/slides/<file>.typ --root .` for slide edits
- `scripts/nbv_qmd_outline.sh --compact` for public navigation checks
- `make check-agent-memory` for `.agents/` or canonical-memory edits

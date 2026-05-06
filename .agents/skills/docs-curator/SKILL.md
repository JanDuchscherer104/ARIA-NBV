---
name: docs-curator
description: Use when ARIA-NBV work changes README, SETUP, public Quarto docs, Typst paper/slides, bibliography, public navigation, docs source-order alignment, or the public/internal docs boundary.
metadata:
  applies_to:
    - "README.md"
    - "SETUP.md"
    - "docs/**"
    - "docs/AGENTS.md"
    - ".agents/references/source_order.md"
  triggers:
    - "docs"
    - "Quarto"
    - "Typst"
    - "bibliography"
    - "public navigation"
    - "public/internal docs boundary"
    - "docs source-order alignment"
  must_read:
    - "docs/AGENTS.md"
    - ".agents/references/source_order.md"
    - ".agents/references/verification_matrix.md"
  verification:
    - "make qmd-frontmatter-check for Quarto docs"
    - "cd docs && quarto render <page.qmd> for changed pages"
    - "make kg-claim-check KG_CLAIM=\"<claim>\" for advisor-facing claims"
---

# Docs Curator

## When To Use

Use this skill for documentation or narrative work that crosses any of:

- `README.md`, setup docs, Quarto pages, Typst paper, or slides
- bibliography and literature references
- generated context routing artifacts
- docs/source-order alignment that explicitly affects public narrative
- public/internal docs boundary decisions

Do not use it for a localized typo fix unless source-of-truth alignment is at risk.
Do not use it for memory-only, backlog-only, KG-consolidation, or internal
operator-reference edits; use `agents-db`, `aria-litkg-memory`, or the owning
skill instead.

## Read First

1. `docs/AGENTS.md`
2. `.agents/references/source_order.md`
3. The source that owns the touched role: seminar paper for implemented
   substrate, thesis roadmap/questions plus memory for active direction, or the
   thesis proposal for advisor proposal wording
4. `.agents/memory/state/GOTCHAS.md` when behavior or workflow claims are involved
5. `.agents/references/agent_memory_templates.md` for debriefs

## Rules

- Treat docs source order as role-split, not as one global narrative.
- Keep public Quarto docs reader-facing; internal agent guidance, generated context, raw scratch history, and OMX runtime notes stay under `.agents/` unless explicitly curated.
- Run litkg claim checks for advisor-facing proposal, roadmap,
  research-question, or literature-synthesis claims.
- Link to canonical state or owning implementation docs instead of repeating long explanations.
- Keep bibliography additions in `docs/references.bib`.
- Update canonical memory only when docs work changes current truth; otherwise
  record `canonical_updates_needed: []` in the debrief.
- Use QMD frontmatter to classify rendered pages:
  `phase: thesis | seminar | archive | generated`,
  `audience: public | advisor | developer | agent`,
  `status: current | planned | scratch | deprecated`, and
  `owner: paper | docs | code | agent | generated | jan`.
- Do not render `audience: agent` pages under `docs/contents/**`; raw internal
  archive and backlog history belongs under `.agents/archive/docs/`.
- Triage docs with these labels: `KEEP_PUBLIC`, `MOVE_TO_AGENTS`,
  `MOVE_TO_PACKAGE_CONTRACT`, `ARCHIVE`, `DELETE`, and
  `GENERATED_UNTRACKED`.
- Keep all retained public QMD files renderable. Only curated archive summaries
  belong under `docs/contents/archive/`.

## Verification

- `cd docs && quarto render contents/thesis/roadmap.qmd contents/thesis/questions.qmd` for roadmap/question edits
- `cd docs && typst compile typst/seminar_paper/main.typ --root .` for paper edits
- `cd docs && typst compile typst/seminar_slides/<file>.typ --root .` for slide edits
- `scripts/nbv_qmd_outline.sh --compact` for public navigation checks
- `make qmd-frontmatter-check` for rendered QMD taxonomy changes
- `make check-agent-memory` for `.agents/` or canonical-memory edits

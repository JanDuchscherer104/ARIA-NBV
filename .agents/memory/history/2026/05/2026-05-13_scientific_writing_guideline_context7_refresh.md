---
id: 2026-05-13_scientific_writing_guideline_context7_refresh
date: 2026-05-13
title: "Scientific Writing Guideline Context7 Refresh"
status: done
topics: [skills, typst, scientific-writing, context7]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/skills/typst-authoring/references/thesis-writing.md
  - .agents/memory/history/2026/05/2026-05-13_scientific_writing_guideline_context7_refresh.md
---

## Task

Updated the ARIA-NBV thesis writing guideline on 2026-05-13 with selected best
practices from Context7 scientific-writing skills while preserving the local
decision that thesis scientific prose belongs under `typst-authoring`.

## Method

The update selectively incorporated outline-to-prose drafting, one-job
paragraphs, IMRAD-as-reader-flow, primary-source citation discipline,
repo-evidence grounding, claim/falsifier scaffolding, evidence audits, and
single-takeaway figure guidance. It explicitly rejected imported defaults that
conflict with ARIA-NBV, including mandatory AI-generated graphical abstracts,
fixed figure quotas, biomedical checklist defaults, generic journal-submission
framing, mandatory external research lookup, and image-generation-first
workflows.

## Verification

Verification performed:

- `python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" .agents/skills/typst-authoring`
  passed.
- `make check-agent-memory` passed.
- `rg -n "graphical abstract|MUST include|CONSORT|STROBE|PRISMA|research-lookup|Nano Banana" .agents/skills/typst-authoring/references/thesis-writing.md`
  returned only the explicit rejected-import bullets.
- `git diff --check -- .agents/skills/typst-authoring/references/thesis-writing.md .agents/memory/history/2026/05/2026-05-13_scientific_writing_guideline_context7_refresh.md`
  passed.
- Manual review against related-work rewrite, methods/results drafting, and
  prose-polish scenarios found the new guidance routes each case to the
  intended section rules, claim scaffold, and style checks.

## Canonical State Impact

No canonical state update is needed. This is a repeatable workflow update in the
active repo-local authoring skill, not a change to thesis direction or public
documentation.

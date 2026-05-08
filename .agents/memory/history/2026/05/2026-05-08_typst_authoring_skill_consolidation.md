---
id: 2026-05-08_typst_authoring_skill_consolidation
date: 2026-05-08
title: "Typst Authoring Skill Consolidation"
status: done
topics: [skills, typst, thesis, writing, scaffold]
confidence: high
canonical_updates_needed: []
---

## Task

Consolidated ARIA-NBV thesis authoring guidance into the repo-local
`typst-authoring` skill. The separate generic scientific-writing skill was
removed from the active skill tree to avoid conflicting biomedical and
journal-submission defaults.

## Method

Selective merge from `.agents/work/aria-nbv-thesis-authoring-skill-handoff`
rather than wholesale copy. Official Typst docs were checked for math
attachments, text operators, project-root imports, and PNG export behavior.

## Result

`typst-authoring` now owns Typst syntax, shared notation, thesis prose,
claim/citation discipline, figures/tables, fixtures, and rendered-page QA.
The helper checker is intentionally advisory because its regexes can match
documented bad examples.

## Verification

Validated through skill validation, fixture compiles, fixture PNG render,
advisory hygiene checks, agent-memory validation, and text diff whitespace
checks.

## Canonical State Impact

No canonical project truth changes were needed. This is a repeatable workflow
update captured in the active skill surface.

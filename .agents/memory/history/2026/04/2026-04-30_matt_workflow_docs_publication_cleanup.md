---
id: 2026-04-30_matt_workflow_docs_publication_cleanup
date: 2026-04-30
title: "Matt Workflow And Docs Publication Cleanup"
status: done
topics: [docs, scaffold, skills, workflow, github-pages]
confidence: high
canonical_updates_needed: []
---

# Matt Workflow and Docs Publication Cleanup

## Summary

Stopped the public Pages workflow from regenerating agent scaffold mirrors into
Quarto content, moved raw archive QMD pages under `.agents/archive/docs/`, and
kept a curated public archive index in `docs/contents/archive/index.qmd`.

## Changes

- Retargeted `scripts/quarto_generate_agent_docs.py` to
  `.agents/generated/agent_scaffold/` and fixed generated GitHub source links
  to the `ARIA-NBV` repository.
- Removed the agent scaffold generation step and `.agents` trigger paths from
  the Pages workflow.
- Added CI-safe glossary generation with `python scripts/glossary_build.py all`
  and a drift check for glossary and notation artifacts.
- Added `scripts/validate_qmd_frontmatter.py` plus
  `make qmd-frontmatter-check` for rendered `docs/contents/**/*.qmd` taxonomy.
- Moved raw `ideas.qmd`, `todos.qmd`, and `repo_structure.qmd` archive pages
  out of the public render tree.
- Hardened ARIA-native Matt workflow guidance inside existing skills instead
  of adding upstream filesystem surfaces.

## Verification

- `python3 scripts/glossary_build.py all`
- `python3 scripts/validate_qmd_frontmatter.py docs/contents`
- `make qmd-frontmatter-check PYTHON_INTERPRETER=python3`
- `python3 -m py_compile scripts/quarto_generate_agent_docs.py scripts/validate_qmd_frontmatter.py scripts/glossary_build.py`

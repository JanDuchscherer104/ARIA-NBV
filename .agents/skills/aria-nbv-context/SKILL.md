---
name: aria-nbv-context
description: Gather deterministic local context for ARIA-NBV by localizing unknown files, symbols, docs, and source families before handing off to narrower skills. Do not use for already-localized one-file edits or KG-backed claim checks.
metadata:
  applies_to:
    - "**"
  triggers:
    - "locate files"
    - "cross-surface context"
    - "where is this implemented"
    - "architecture map"
    - "source family"
  must_read:
    - "AGENTS.md"
    - ".agents/references/source_order.md"
  verification:
    - "make context when generated context is stale or missing"
---

# Aria NBV Context

Use this skill as the local discovery layer. It should identify the smallest
relevant set of files, then hand off to a narrower implementation, docs, KG, or
diagnostic workflow.

Use `aria-litkg-memory` instead when the task needs KG-backed retrieval,
source-backed claim checking, active backlog routing, or consolidation.

## Retrieval Ladder

1. Read `AGENTS.md` and `.agents/references/source_order.md`.
2. Use `docs/_generated/context/source_index.md` for source-family routing only
   when it exists; run `make context` first only if the generated index is
   needed and missing or stale.
3. Open the nearest nested `AGENTS.md` once the surface is known.
4. Use source-specific outline tools before broad raw reads:
   - Quarto: `scripts/nbv_qmd_outline.sh --compact`
   - Typst paper: `scripts/nbv_typst_includes.py --paper --mode outline`
   - Literature: `scripts/nbv_literature_index.sh`
   - Code: `scripts/nbv_get_context.sh modules|contracts|match <term>`
5. Use targeted `rg` inside the narrowed file set.
6. Refresh with `make context` only when lightweight generated context is stale
   or missing.
7. Use `make context-heavy` or specific heavy context targets only after lighter
   discovery fails.

## Do Not Use When

- The user already named the exact file to edit or review.
- The task is a localized code change inside one module.
- The task needs KG authority/freshness, claim-checking, or memory
  consolidation.
- The task has a concrete failure command; use `diagnose-aria`.

## Zoom-Out Output

When asked to map a surface, return:

- domain term and glossary anchor when one exists
- owning package/module and nearest `AGENTS.md`
- main callers and data contracts
- relevant tests or render checks
- docs/memory surfaces likely to need updates
- open risks or missing context

## References

- `references/context_map.md` for non-obvious concept-to-source routing.
- `.agents/references/source_order.md` for role-specific truth ownership.

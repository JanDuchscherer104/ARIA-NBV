---
name: typst-authoring
description: Use for ARIA-NBV Typst proposal/thesis authoring, shared notation, scientific prose, citations, figures/tables, and compile/render QA.
metadata:
  mode: implementation
  not_when:
    - "pure Quarto prose without Typst or thesis scientific-writing concerns"
    - "a concrete Typst/Quarto traceback or failing command owns the task"
    - "a broad advisor-facing thesis-scope decision is unresolved"
  handoff_to:
    - "docs-curator for public Quarto navigation or docs-boundary edits"
    - "diagnose-aria for concrete build failures or suspicious rendered output"
    - "plan-grill for ambiguous advisor-facing research-contract decisions"
  evidence_required:
    - "nearest docs guidance and target Typst imports"
    - "shared notation/glossary check for new symbols or equations"
    - "compile and rendered-page inspection for non-trivial visual/math edits"
  applies_to:
    - "docs/typst/**"
    - ".agents/skills/typst-authoring/**"
  triggers:
    - "Typst"
    - "proposal.typ"
    - "thesis Typst"
    - "shared symbols or equations"
    - "scientific prose in thesis/proposal"
  must_read:
    - "AGENTS.md"
    - "docs/AGENTS.md"
    - ".agents/references/source_order.md"
  verification:
    - "skill quick_validate.py for skill edits"
    - "focused typst compile plus PNG render for document edits"
    - "make check-agent-memory when agent guidance changes"
---

# ARIA-NBV Typst + Thesis Authoring

This is the active guardrail for ARIA-NBV proposal and thesis writing in
Typst. Treat Typst correctness, shared notation, scientific prose, citations,
figures, tables, and visual QA as one workflow.

## Use When

- Editing `.typ` proposal, paper, thesis, or slide sources.
- Adding or revising equations, symbols, glossary-backed terms, citations,
  figures, tables, captions, or labels.
- Polishing advisor-facing thesis prose that must become evidence-backed
  paragraphs rather than bullet scaffolding.
- Updating this skill's fixtures, references, or helper scripts.

## Do Not Use When

- The task is only Quarto navigation/frontmatter; use `docs-curator`.
- A build failure already has a concrete traceback; use `diagnose-aria`.
- The research contract or thesis scope is still ambiguous; use `plan-grill`.

## Rules

1. Inspect local context first: nearest `AGENTS.md`, target imports, adjacent
   sections, bibliography style, labels, and `docs/typst/shared/`.
2. Use shared ARIA-NBV notation before inventing local symbols:
   `macros.typ`, `symbols.typ`, `equations.typ`, `glossary.typ`, `terms.typ`,
   and `math.typ`.
3. Add recurring symbols/equations to `docs/typst/shared` first, then use them
   from the document. Do not duplicate notation inline across sections.
4. Use typed notation: bold data vectors, matrices, tensors, feature fields,
   embeddings, images/depth maps, point-cloud collections, and voxel tensors;
   keep abstract sets/operators unbolded unless the shared symbol says
   otherwise.
5. Handle Typst attachments defensively. After `_` or `^`, insert a space
   before following arguments when they should not be captured, and group full
   calls before output indexing: `(op("Transformer")_theta (bold(X)_t))_i`.
6. Classify every scientific claim as definition, literature claim,
   implementation fact, design decision, empirical result, limitation, or
   hypothesis. If it cannot be classified, rewrite it.
7. Final proposal/thesis prose should be flowing paragraphs unless the template
   explicitly asks for lists. Bullet outlines are planning scaffolds only.
8. Figures and tables must be evidence: use `#figure(...)`, stable labels,
   explicit sizing, concise captions, and prose that states the claim the
   visual supports.
9. Use Typst symbols, math shorthands, or shared macros instead of raw Unicode
   glyphs or LaTeX commands.
10. Compile and inspect rendered pages for equations, figures, tables,
    captions, layout changes, and multi-paragraph thesis prose.

## Workflow

1. Read the target file and the relevant on-demand references below.
2. If notation changes, check `docs/typst/shared` and update the shared module
   before using the symbol in thesis text.
3. If prose changes, draft claims/evidence first, then convert to paragraphs.
4. If figures or Mermaid assets change, render them locally before including
   them in Typst.
5. Compile the document or fixture, render affected pages to PNG, inspect
   visually, then fix and repeat.
6. Report the exact compile/render commands run and any skipped checks.

## References

- `issues.md` - regression checklist with bad/good examples.
- `references/aria-nbv-notation.md` - project notation and shared imports.
- `references/math-attachments.md` - Typst attachment and operator patterns.
- `references/thesis-writing.md` - thesis prose, ABT/CARS framing, anti-fluff.
- `references/claim-citation-discipline.md` - claim taxonomy and evidence gates.
- `references/figures-tables.md` - figure, table, caption, and Mermaid policy.
- `references/workflow.md` - compile, render, inspect, and final QA loop.
- `references/context7-research-queries.md` - targeted docs/skill refreshes.
- Existing Typst references for data loading, scripting, layout, packages, and
  slides remain available on demand.

## Fast Commands

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" .agents/skills/typst-authoring
typst compile .agents/skills/typst-authoring/assets/fixtures/shared-notation.typ /tmp/shared-notation.pdf --root .
.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/proposal.typ -o /tmp/proposal-pages --root docs --pages 1-3 --ppi 300
.agents/skills/typst-authoring/scripts/hygiene_checks.sh .agents/skills/typst-authoring docs/typst/thesis/sections/proposal
```

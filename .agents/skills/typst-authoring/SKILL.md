---
name: typst-authoring
description: Use for ARIA-NBV Typst proposal/thesis authoring, shared notation, scientific prose, citations, figures/tables, Mermaid inclusion, and compile/render QA.
metadata:
  mode: implementation
  not_when:
    - "pure Quarto navigation/frontmatter without Typst or thesis scientific-writing concerns"
    - "a systemic, CI-specific, or persistent docs build failure owns the task"
    - "a broad advisor-facing thesis-scope decision is unresolved"
  handoff_to:
    - "docs-curator for public Quarto navigation or docs-boundary edits"
    - "diagnose-aria for systemic build failures or suspicious rendered output that persists after the Typst loop"
    - "plan-grill for ambiguous advisor-facing research-contract decisions"
  evidence_required:
    - "nearest docs guidance and target Typst imports"
    - "shared notation/glossary check for new symbols, equations, or durable terms"
    - "claim/citation check for advisor-facing literature or thesis claims"
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
    - "skill quick_validate.py when available for skill edits"
    - "make check-agent-memory when agent guidance changes"
    - "focused Typst compile plus PNG render for document edits"
---

# ARIA-NBV Typst + Thesis Authoring

This is the repo-local guardrail for ARIA-NBV proposal and thesis writing in
Typst. Treat Typst correctness, shared notation, scientific prose, citations,
figures, tables, Mermaid inclusion, and visual QA as one workflow.

## Use When

- Editing `.typ` proposal, paper, thesis, or slide sources.
- Adding or revising equations, symbols, glossary-backed terms, citations,
  figures, tables, captions, labels, or Mermaid-derived figures.
- Polishing advisor-facing thesis prose into evidence-backed paragraphs.
- Updating this skill's fixtures, references, or helper scripts.
- Fixing ordinary Typst syntax, math attachment, import, citation, figure,
  table, label, or rendered-page issues.

## Do Not Use When

- The task is only Quarto navigation/frontmatter; use `docs-curator`.
- A failure is systemic, CI-specific, multi-surface, or persists after the
  compile/render loop; use `diagnose-aria`.
- The research contract or thesis scope is still ambiguous; use `plan-grill`.

## Task Modes

- `notation-edit`: read `references/aria-nbv-notation.md`,
  `references/math-attachments.md`, `references/notation-migration.md`, and
  `issues.md`; update shared modules before document-local use; compile a
  fixture or affected document.
- `prose-draft`: read `references/thesis-writing.md`,
  `references/thesis-section-contracts.md`, and
  `references/claim-citation-discipline.md`; build a claim ledger for
  non-trivial sections before writing paragraphs.
- `prose-polish`: preserve claim strength and citations; remove filler,
  overclaiming, and local terminology drift without changing evidence.
- `claim-check`: classify claims, verify citations/evidence, and run
  `make kg-claim-check KG_CLAIM='...'` for advisor-facing literature or
  thesis claims.
- `figure-table`: read `references/figures-tables.md`; keep source assets,
  labels, sizing, captions, and final page rendering aligned.
- `visual-qa`: compile, render affected pages to PNG, inspect equations and
  layout, run hygiene checks, and report commands plus skipped checks.

## Rules

1. Inspect local context first: nearest `AGENTS.md`, target imports, adjacent
   sections, bibliography style, labels, and `docs/typst/shared/`.
2. Use shared ARIA-NBV notation before inventing local symbols:
   `macros.typ`, `symbols.typ`, `equations.typ`, `glossary.typ`, `terms.typ`,
   and `math.typ`.
3. Add recurring symbols/equations to `docs/typst/shared` first, then use them
   from the document. Do not duplicate notation inline across sections.
4. Use typed notation: `cal(...)` for abstract sets, spaces, candidate sets,
   point sets, meshes, and geometric collections; `bold(...)` only for
   coordinate vectors, matrices, tensors, feature fields, embeddings,
   images/depth maps, voxel tensors, and implementation arrays. Do not write
   `bold(cal(...))`.
5. Keep finite-candidate and value notation disjoint: candidate sets are
   `cal(Q)_t`, candidate poses are `q_(t,i)`, candidate feature tables are
   `bold(X)_t^"cand"`, and `Q_H` / `Q_(H,theta)` are value functions only.
6. Keep abstract states non-bold: `s_t^"obs"`, `s_t^"cf0"`,
   `s_t^"oracle"`. Use `bold(h)_t` or `bold(u)_(t,i)` for learned state and
   candidate embeddings.
7. ARIA-NBV thesis RRI equations use point-mesh error `D`, directional
   components `D_(P -> M)` and `D_(M -> P)`, and target error `Delta_t^e`.
   Do not reintroduce generic `CD(...)` or `cal(A)` / `cal(C)` as thesis-core
   component notation.
8. Handle Typst attachments defensively. After `_` or `^`, insert a space
   before following arguments when they should not be captured, and group full
   calls before output indexing: `(op("Transformer")_theta (bold(X)_t))_i`.
9. Classify every scientific claim as definition, literature claim,
   implementation fact, design decision, empirical result, limitation, or
   hypothesis. If it cannot be classified, rewrite it.
10. For advisor-facing literature or thesis claims, run `make kg-claim-check`
   and downgrade, cite, or mark as hypothesis when evidence is weak.
11. If a new durable acronym, term, or recurring noun phrase is introduced,
   update `docs/typst/shared/glossary.typ` and run `make glossary`.
12. Final proposal/thesis prose should be flowing paragraphs unless the
   template explicitly asks for lists. Bullet outlines are planning scaffolds.
13. In `.typ` prose, do not use source line breaks for visual wrapping; VSCode
    wraps long lines automatically. Insert line breaks only where the rendered
    document should have a paragraph, block, list item, table row, equation
    structure, or other intentional Typst boundary.
14. Figures and tables must be evidence: use `#figure(...)`, stable labels,
    explicit sizing, concise captions, and prose that states the claim the
    visual supports.
15. Use Typst symbols, math shorthands, or shared macros instead of raw
    Unicode glyphs or LaTeX commands.
16. Compile and inspect rendered pages for equations, figures, tables,
    captions, layout changes, and multi-paragraph thesis prose.

## Workflow

1. Choose the task mode and read only its required references.
2. If notation changes, check `docs/typst/shared` and update the shared module
   before using the symbol in thesis text.
3. If prose changes, draft claims/evidence first, then convert to paragraphs.
4. If figures or Mermaid assets change, render them locally before inclusion.
5. Compile the document or fixture, render affected pages to PNG, inspect
   visually, then fix and repeat.
6. Report exact compile/render/check commands and any skipped checks.

## Reference Map

| Need | Read |
| --- | --- |
| Regression examples | `issues.md` |
| ARIA notation/glossary | `references/aria-nbv-notation.md` |
| Typst math attachments | `references/math-attachments.md` |
| Shared-notation migration | `references/notation-migration.md` |
| Thesis prose | `references/thesis-writing.md` |
| Section acceptance | `references/thesis-section-contracts.md` |
| Claims/citations | `references/claim-citation-discipline.md` |
| Figures/tables/Mermaid | `references/figures-tables.md` |
| Compile/render loop | `references/workflow.md` |
| External refresh queries | `references/external-research.md` |
| Data tables | `references/data-loading.md` |
| Typst scripting/modules | `references/scripting.md` |
| Layout/page grids | `references/layout.md` |
| Raw glyph avoidance | `references/typst-symbols.md` |
| Typst packages | `references/packages/index.md` |
| Slides | `references/slides.md` |

## Fast Commands

```bash
# Skill validation when the local validator exists.
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" .agents/skills/typst-authoring

# Required for repo-local skill or memory/guidance edits.
make check-agent-memory

# Primary document builds.
make proposal-pdf
make thesis-pdf

# Manual fallback from repo root.
cd docs && typst compile typst/thesis/proposal.typ /tmp/proposal.pdf --root .

# Fixtures and visual checks.
typst compile .agents/skills/typst-authoring/assets/fixtures/shared-notation.typ /tmp/shared-notation.pdf --root .
.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/proposal.typ -o /tmp/proposal-pages --root docs --pages 1-3 --ppi 300
.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/sections/proposal
```

# Typst Edit, Compile, Render, Inspect Loop

Use this loop for non-trivial edits: equations, figures, tables, layout,
captions, bibliography/cross-references, or multi-paragraph thesis prose.

## 1. Inspect Local Context

```bash
make context-typst-includes TYPST_INCLUDES_ARGS='--paper --mode includes'
make context-typst-outline TYPST_OUTLINE_ARGS='--paper --mode outline'
```

Then read the target file, adjacent sections, and relevant files under
`docs/typst/shared`.

## 2. Isolate Fragile Objects

For a complex equation, table, or figure, create or update a small fixture
first. This reduces noise and makes visual errors obvious.

## 3. Compile

```bash
typst compile docs/typst/thesis/proposal.typ /tmp/proposal.pdf --root docs
```

Use the correct root for the document. If compiling files under
`.agents/skills`, use `--root .`.

## 4. Render Affected Pages

```bash
.agents/skills/typst-authoring/scripts/render_png.sh \
  -i docs/typst/thesis/proposal.typ \
  -o /tmp/proposal-pages \
  --root docs \
  --pages 1-4 \
  --ppi 300
```

Use `--ppi 600` for detailed equation/figure inspection.

## 5. Inspect Visually

Check attachment scope after `_` / `^`, bolding and symbol consistency, line
breaks and equation overflow, figure scale/cropping, caption clarity, table
alignment, cross-reference output, and awkward page breaks.

## 6. Fix And Repeat

A clean compile alone is insufficient when the change affects rendering. Repeat
until the affected pages are visually clean.

## 7. Final Hygiene

```bash
.agents/skills/typst-authoring/scripts/hygiene_checks.sh .agents/skills/typst-authoring docs/typst/thesis
make check-agent-memory
git diff --check
```

Report exactly what was checked. If a command cannot run, say why.

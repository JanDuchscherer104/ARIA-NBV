---
id: 2026-05-06_core_math_lookup_render_fix
date: 2026-05-06
title: "Core Math Lookup Render Fix"
status: done
topics: [docs, glossary, quarto, math]
confidence: high
canonical_updates_needed: []
files_touched:
  - scripts/glossary_build.py
  - docs/contents/glossary.qmd
---

## Task

Fix the public Quarto glossary Core Math Lookup section, where generated notation
rendered as stripped placeholder text such as `(_{})` and `((q)=)`.

## Method

Verified that the generated QMD had TeX notation embedded inside raw HTML spans
with `\(...\)` delimiters, and that Quarto/Pandoc mangled those spans before
MathJax could typeset them. Updated the glossary generator to emit `$...$`
inline math inside the existing raw HTML layout, then regenerated the glossary
with `make glossary`.

## Verification

- `make glossary`
- `cd docs && quarto render contents/glossary.qmd`
- Checked `docs/_site/contents/glossary.html` for proper
  `class="math inline"` notation around `\mathcal` and `\mathrm{RRI}`.
- Checked that the known broken lookup pattern no longer appears.
- `make qmd-frontmatter-check`
- `git diff --check -- scripts/glossary_build.py docs/contents/glossary.qmd`

## Canonical State Impact

No durable state update needed. The existing decision that the public glossary is
generated from shared Typst glossary/notation sources remains unchanged.

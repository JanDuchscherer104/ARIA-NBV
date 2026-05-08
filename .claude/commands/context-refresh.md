---
description: Refresh the lightweight scaffold context (make context).
allowed-tools: Bash(make context), Bash(make context-contracts)
argument-hint: "[contracts]"
---

If $ARGUMENTS is "contracts", run `make context-contracts`. Otherwise run
`make context` to refresh `docs/_generated/context/source_index.md`,
`literature_index.md`, and `data_contracts.md`.

Use the source-specific outline tools before broad raw reads:
- Quarto: `.agents/skills/aria-nbv-context/scripts/nbv_qmd_outline.sh --compact`
- Typst: `.agents/skills/aria-nbv-context/scripts/nbv_typst_includes.py --paper --mode outline`
- Literature: `.agents/skills/aria-nbv-context/scripts/nbv_literature_index.sh`
- Code/contracts: `.agents/skills/aria-nbv-context/scripts/nbv_get_context.sh modules|contracts|match <term>`

`make context-heavy`, `context-uml`, `context-docstrings`, and `context-tree`
are explicit fallback tools — use them only for architecture or refactor work.

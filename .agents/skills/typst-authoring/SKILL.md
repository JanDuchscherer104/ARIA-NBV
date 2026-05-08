---
name: typst-authoring
description: Create or edit Typst documents (papers, reports, slides, figures, tables, math, citations). Use when working in .typ files, building layouts, or loading data (CSV/JSON) for tables and plots.
---

# Typst Authoring

## Overview

Author or revise Typst documents with consistent layout, labels, and data handling.
Use this workflow to keep edits aligned with local templates and official Typst docs.

## Workflow

1) Discover local style constraints:
   - Look for AGENTS/STYLE files, templates, or existing `.typ` sections nearby.
   - Mirror layout, spacing, and caption conventions from adjacent content.

2) Pull Typst docs via Context7:
   - Query `/websites/typst_app` for additional context.
   - `$typst-authoring` provides a good overview of DOs & DON'Ts as well as further context indices on good typst usage.

3) Follow the "compile -> inspect -> fix loop" for non-trivial changes as per $typst-authoring

## Data Structures (Arrays + Dictionaries)

Use this mini-workflow when shaping data for tables or templates:

1) Load Context7 docs for arrays + dictionaries (see `references/context7-queries.md`).
2) Choose the transform path:
   - **Array -> Array:** `map`, `filter`, `slice`, `reduce`, `join`
   - **Array -> Dictionary:** `array.to-dict()` for pairs
   - **Dictionary access:** `.at("key")` or `.key` for static keys
3) Flatten for tables and preserve ordering for reproducible outputs.

## References inside $typst-authoring

- `references/style-guide.md`: tone, labels, and layout conventions.
- `references/context7-queries.md`: required/optional Typst queries.
- `references/typst-essentials.md`: minimal Typst directive/label/layout cheat sheet.
- `references/workflow.md`: compile/inspect checklist.
- `references/typst-docs-notes.md`: Context7-aligned notes with doc sections.
- `references/typst-symbols.md`: symbol usage rules and namespaces (no Unicode glyphs).
- `references/layout.md`: layout primitives and patterns (aligned to Typst layout docs).
- `references/data-loading.md`: CSV/TOML/JSON patterns and data-loading caveats.
- `references/typst-data-structures.md`: array/dictionary patterns for tables and templates.
- `references/scripting.md`: scripting, control flow, modules, and blocks.
- `references/slides.md`: slide creation with our custom definitely-not-isec template and Touying essentials.
- `references/packages/`: package-specific notes (`fletcher.md`, `booktabs.md`) and template `.typ` files.

---
name: aria-nbv-context
description: Gather targeted context for Aria-NBV from Quarto docs (docs/**/*.qmd), Typst paper/slides (docs/typst/**), LaTeX literature (literature/**), and source code (oracle_rri/**). Use when the user asks for background, citations, paper/slide updates, or when you need to locate specific technical details across docs and code.
---

# Aria NBV Context

## Overview
Use this skill to rapidly collect *specific* context from NBV docs, Typst paper/slides, literature sources, and code. Start broad (context snapshot), then drill down with targeted lookups and search.

## Quick Start (default)
1. Run `scripts/nbv_context_index.sh` to generate a source index at `.codex/context_sources_index.md`.
2. Run `make context` (writes to `.codex/codex_make_context.md` and embeds the source index).
3. Use `rg` on the index file to pick exact files, then `rg` within those sources.
4. Use the scripts below to list/outline relevant sources, then drill into files.

## Source index (first step)
`scripts/nbv_context_index.sh` writes a single Markdown index of *all* source pools:

- Quarto docs (`docs/**/*.qmd`)
- Typst paper/slides/shared (`docs/typst/**`)
- Literature LaTeX/Bib (`literature/**`)
- Python source (`oracle_rri/**`)

This index is the entry point for compositional context gathering: search it first to find the smallest set of files, then `rg` inside those files.

## `.codex/codex_make_context.md` structure (what to search)
The snapshot is a single Markdown file with stable section headers (includes the full source index):

- **Header + Contents**: title, generated timestamp, and `## Contents`
- **Section 0**: `## 0) Source index (all context pools)`
- **Section 1**: `## 1) Environment` (python/venv info)
- **Section 2**: `## 2) Mermaid UML (oracle_rri)` (UML class diagram)
- **Section 3**: `## 3) Class docstrings (oracle_rri)` (full AST docs)
- **Section 4**: `## 4) Directory tree (oracle_rri)` (tree output)

Search examples:
- `rg -n "^## 2\\) Mermaid UML" .codex/codex_make_context.md`
- `rg -n "VinModelV3|CandidateDepthRenderer" .codex/codex_make_context.md`
- `rg -n "^## 4\\) Directory tree" .codex/codex_make_context.md`

## Context Sources

### 1) Quarto docs (docs/**/*.qmd)
- Preferred entry points:
  - `docs/index.qmd`
  - `docs/contents/todos.qmd`
- Use `make context-qmd-tree` for a high-level map.
- Use `scripts/nbv_qmd_outline.sh` (wrapper) or `.codex/skills/aria-nbv-context/scripts/nbv_qmd_outline.sh` to list headings across docs with nested ordering.

### 2) Typst paper (docs/typst/paper)
- Start with `docs/typst/paper/main.typ`.
- Use `scripts/nbv_typst_includes.py` (wrapper) or `.codex/skills/aria-nbv-context/scripts/nbv_typst_includes.py` to expand include graphs with nested section headings.
- Use `--mode includes` for include-only output.
- Check `docs/typst/shared/macros.typ` and `docs/typst/paper/macros.typ` for symbols.

### 3) Typst slides (docs/typst/slides)
- List slide sources via `scripts/nbv_typst_includes.py` (wrapper) or `.codex/skills/aria-nbv-context/scripts/nbv_typst_includes.py` (paper + slides).
- Open only the slide file that matches the topic.

### 4) Literature LaTeX (literature/**)
- Use `scripts/nbv_literature_search.sh "<query>"` (wrapper) or `.codex/skills/aria-nbv-context/scripts/nbv_literature_search.sh "<query>"` for targeted grep.
- Focus on `.tex` and `.bib` sources; avoid loading large PDFs.

### 5) Source code (oracle_rri/**)
- Use `scripts/nbv_get_context.sh packages` (wrapper) or `.codex/skills/aria-nbv-context/scripts/nbv_get_context.sh packages` (or `classes`) to get AST-based summaries.
- For focused code search, use `rg` or the code-index MCP tools.

## Targeted Search (rg + code-index)
Use `rg` for fast, precise text search across sources. When searching code, the
`code-index` MCP tools can provide symbol summaries and definitions without
opening full files. Prefer the scripts in this skill for outlines and include
graphs, then use `rg` to narrow further.

Suggested entry-point search:
- `rg -n "<term>" .codex/context_sources_index.md`

Suggested patterns for `.codex/codex_make_context.md`:
- `rg -n "^## 0\\) Source index" .codex/codex_make_context.md`
- `rg -n "^## 2\\) Mermaid UML" .codex/codex_make_context.md`
- `rg -n "^## 3\\) Class docstrings" .codex/codex_make_context.md`
- `rg -n "VinModelV3|CandidateDepthRenderer" .codex/codex_make_context.md`

Cheat sheet (common lookups):
```bash
# Data views and snippet types
rg -n "EfmSnippetView|EfmCameraView|EfmPointsView" .codex/codex_make_context.md

# Candidate generation + pose sampling
rg -n "pose_generation\\.candidate_generation|CandidateSamplingResult" .codex/codex_make_context.md

# Rendering pipeline (depth + point clouds)
rg -n "rendering\\.candidate_depth_renderer|CandidateDepthRenderer" .codex/codex_make_context.md

# Oracle labeling + cache
rg -n "pipelines\\.oracle_rri_labeler|OracleRriLabeler|OracleRriCache" .codex/codex_make_context.md

# VIN models + encoders
rg -n "vin\\.model_v3|VinModelV3|pose_encoders|traj_encoder" .codex/codex_make_context.md
```

## Bundled Scripts
- Scripts are self-rooting and can be run from any working directory.
- Convenience wrappers live in `scripts/` and delegate to the skill scripts.
- `.codex/skills/aria-nbv-context/scripts/nbv_context_index.sh`:
  - writes `.codex/context_sources_index.md` (full source index).
- `.codex/skills/aria-nbv-context/scripts/nbv_get_context.sh`:
  - wrapper around `oracle_rri/scripts/get_context.py` with the correct root.
- `.codex/skills/aria-nbv-context/scripts/nbv_qmd_outline.sh`:
  - list headings in all `docs/**/*.qmd`.
- `.codex/skills/aria-nbv-context/scripts/nbv_typst_includes.py`:
  - extract `#include` targets from Typst paper/slides.
- `.codex/skills/aria-nbv-context/scripts/nbv_literature_search.sh`:
  - grep across `literature/**` LaTeX/Bib sources.

## References
- `references/context_map.md` provides topic-to-file routing for fast lookup.

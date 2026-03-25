---
id: 2026-01-20_typst_compile_root_dot_fix_2026-01-20
date: 2026-01-20
title: "Typst Compile Root Dot Fix 2026 01 20"
status: legacy-imported
topics: [typst, compile, root, dot, 2026]
source_legacy_path: ".codex/typst_compile_root_dot_fix_2026-01-20.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Typst paper compile path fixes (paper + `docs/typst`)

## Request
Make these commands work without path errors:

- `cd docs/typst/paper && typst compile main.typ`
- `cd docs/typst && typst compile paper/main.typ --root .`

## Findings
- Typst treats paths starting with `/` as rooted at `--root` (project root).
- Typst blocks reading files outside the project root (“access denied”), so `../..` paths break when compiling from `docs/typst/paper` with the default root.

## Changes
- Made the paper self-contained under `docs/typst/paper/`:
  - `docs/typst/paper/macros.typ` (paper-local macros)
  - `docs/typst/paper/references.bib` (copy of `docs/references.bib`)
  - `docs/typst/paper/figures/` (copied the figure files referenced by the paper)
- Updated paper sections to avoid `../..` paths:
  - Use `#import "../macros.typ": *` from section files.
  - Use `image("/figures/...")` so assets resolve inside the paper root.
- Kept `cd docs/typst && typst compile paper/main.typ --root .` working by mirroring:
  - `docs/typst/references.bib`
  - `docs/typst/figures/`
- Deleted `docs/typst/paper/compile.sh` (user requested no custom wrapper).

## Validation
- `cd docs/typst/paper && typst compile main.typ` succeeds.
- `cd docs/typst && typst compile paper/main.typ --root .` succeeds.

## Follow-ups / Gotchas
- `docs/references.bib` remains the source of truth; if it changes, refresh the copies (`docs/typst/references.bib`, `docs/typst/paper/references.bib`).
- If new figures are referenced in the Typst paper, copy them into `docs/typst/paper/figures/` (and into `docs/typst/figures/` if you still compile with `--root .` from `docs/typst`).

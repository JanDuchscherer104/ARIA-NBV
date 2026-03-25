---
id: 2026-01-20_typst_compile_no_root_fix_2026-01-20
date: 2026-01-20
title: "Typst Compile No Root Fix 2026 01 20"
status: legacy-imported
topics: [typst, compile, no, root, 2026]
source_legacy_path: ".codex/typst_compile_no_root_fix_2026-01-20.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Typst paper compile fix (`typst compile main.typ` in `docs/typst/paper`)

## Request
Make this work from the paper directory without `--root`:

`cd docs/typst/paper && typst compile main.typ`

## Problem
Typst forbids reading files outside the project root. With project root = `docs/typst/paper`, any `../..` paths from `sections/*.typ` (e.g. `../../figures/...`) fail with “access denied”.

## Changes
- Updated all paper sections to:
  - import macros via `#import "../macros.typ": *` (stays inside project root)
  - load figures via absolute `/figures/...` paths
- Mirrored the required figures into `docs/typst/paper/figures/` so `/figures/...` resolves when the paper dir is the root.

## Validation
- `cd docs/typst/paper && typst compile main.typ` succeeds.

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


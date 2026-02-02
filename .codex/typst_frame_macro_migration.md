# Typst frame macro migration (`fr_*` → `symb.frame.*`)

## Goal
Replace legacy Typst frame helpers (`fr_world`, `fr_rig_ref`, `fr_cam`, `fr_voxel`) with the canonical `symb.frame.*` symbols defined in `docs/typst/shared/macros.typ`.

## Mapping used
- `fr_world` → `symb.frame.w`
- `fr_rig_ref` → `symb.frame.r`
- `fr_cam` → `symb.frame.cq`
- `fr_voxel` → `symb.frame.v`

## Changes
- Updated paper + slides sources to use `#symb.frame.*` and `#T(symb.frame.A, symb.frame.B)` instead of `fr_*`.
- Removed the legacy `fr_*` `#let` bindings from `docs/typst/shared/macros.typ` to prevent accidental reuse.

## Files touched
- `docs/typst/paper/sections/05-coordinate-conventions.typ`
- `docs/typst/paper/sections/12d-appendix-vin-v2-details.typ`
- `docs/typst/paper/sections/12f-appendix-pose-frames.typ`
- `docs/typst/shared/macros.typ`
- `docs/typst/slides/slides_4.typ`
- `docs/typst/slides/template.typ`

## Validation
- `typst compile --root docs docs/typst/paper/sections/12f-appendix-pose-frames.typ /tmp/appendix-pose-frames.pdf`
- `typst compile --root docs docs/typst/slides/slides_4.typ /tmp/slides_4.pdf`
- `docs/typst/paper/main.typ` currently fails to compile due to a pre-existing Typst error in `docs/typst/paper/sections/12b-appendix-extra.typ` (unrelated to this change).

## Follow-ups
- Fix `docs/typst/paper/sections/12b-appendix-extra.typ` so the full paper compiles again.

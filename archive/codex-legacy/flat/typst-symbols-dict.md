## Summary
- Replaced `sym_*` macros with a single dictionary `s` (dot access) in Typst.

## Changes
- `docs/typst/shared/macros.typ`: defined `#let s = (...)` with all symbol entries.
- Converted all `sym_*` usages in `docs/typst/**/*.typ` to `s.<key>`.
- Wrapped scripted uses as `#(s.key)_...` where needed to avoid spacing issues.

## Notes
- Access is now `#s.key` in math (e.g., `$#s.points$`), with `#(s.key)_t` for scripts.
- `s.W` is reserved for weight matrices; width dimension is `s.Wdim`.

# Typst Unicode Cleanup Notes

This repo aims to keep Typst sources ASCII-only (no Unicode characters in `*.typ`)
while still typesetting symbols via Typst's built-in symbol system.

## How to scan for Unicode

```bash
rg -nP "[^\\x00-\\x7F]" docs/typst -g'*.typ'
```

## Common replacements

- Right arrow: `#sym.arrow.r` (markup), `arrow.r` (math)
- Left arrow: `#sym.arrow.l` (markup), `arrow.l` (math)
- Left-right arrow: `#sym.arrow.l.r` (markup), `arrow.l.r` (math)
- Degree sign: `#sym.degree` (markup), `degree` (math)
- Elementwise / circled-dot product (⊙): `dot.o` (math)
- Approx (≈): `approx` (math) or `~=` in code/text

## Quick sanity compile

```bash
typst compile --root docs docs/typst/paper/main.typ .codex/_render/paper.pdf --diagnostic-format short
```


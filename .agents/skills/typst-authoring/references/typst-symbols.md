# Typst Symbols (Use `#sym` or math shorthands — never raw Unicode)

## Golden rule
- **Do not insert Unicode glyphs directly.** Always use Typst symbols or shorthands.
- **Markup mode:** use `#sym.<namespace>` (or `#emoji.<namespace>` for emoji).
- **Math mode:** you can omit `#sym.` and use the symbol name directly, or use shorthands like `->`, `=>`, `>=`.

## Core namespaces
- `sym` — named symbols for markup and math.
- `emoji` — named emoji (avoid for academic text unless needed).
- `math` — math-only names; includes `dif` / `Dif` (for differentials).

## Common symbol categories (look up via Context7)
Use these categories as a guide; query Context7 to get exact names:
- **Arrows & directions:** query `sym arrow relation`
- **Relations & comparisons:** query `math symbols relation`
- **Logic:** query `math symbols logic`
- **Set/number theory:** query `math symbols set`
- **Greek letters:** query `math greek`
- **Operators & delimiters:** query `math symbols`

## Common patterns (examples)
- Arrows (markup): `#sym.arrow` or `#sym.arrow.r`  
- Relations (math): `$x >= y$`, `$x -> y$`, `$x => y$`  
- Differential (math): `$dif x$`  
- Differential (markup): `math.dif`

## Shorthands (safe in math mode)
Use shorthands in math mode and avoid raw Unicode glyphs in markup:
- `->`, `=>`, `<=`, `>=`, `!=`, `~` (non-printing in markup).
- If a shorthand conflicts with text, escape characters to disable it.

## When to look up symbol names
If unsure about a symbol, query Context7 and use the `sym`/`math` name:
- **Query:** `foundations symbol sym emoji`
- **Query:** `math symbols shorthands dif Dif`

## Notes
- Named symbols can be used without the `#sym.` prefix inside `$...$`.
- `dif`/`Dif` are special math names that also affect spacing and style.

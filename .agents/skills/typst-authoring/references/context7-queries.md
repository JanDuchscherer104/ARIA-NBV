# Context7 Queries (Typst)

Use `/websites/typst_app` and keep queries narrow.

## Required
- `math symbols figure`

## Common Authoring Tasks
- **Figures/labels:** `figure caption label reference`
- **Tables:** `table table.header align stroke inset`
- **Layout:** `grid stack wrap-content`
- **Show rules:** `show set text`
- **Layout (core):** `layout align block box columns grid stack place pad page pagebreak measure`
- **Export PNG:** `export png ppi pages`
- **Slides (Touying):** `touying slide pause reveal speaker-note composer appendix`

## Data Loading & Scripting
- **CSV:** `csv data-loading row-type dictionary`
- **JSON:** `json data-loading`
- **TOML:** `toml data-loading`
- **Scripting (core):** `scripting let blocks content destructuring`
- **Scripting (control):** `if for while break continue return`
- **Scripting (modules):** `include import module`
- **Scripting (style/context):** `set show context`
- **Scripting (helpers):** `calc round map slice filter`
- **Symbols (core):** `foundations symbol sym emoji`
- **Symbols (math):** `math symbols shorthands dif Dif`
- **Symbols (arrows/relations):** `sym arrow relation`
- **Symbols (logic/sets/greek):** `math symbols logic set greek`

## Data Structures
- **Array core:** `array foundations array`
- **Dictionary core:** `dictionary foundations dictionary`
- **Array ops:** `array map filter slice join reduce`
- **Array to dict:** `array to-dict`
- **Dictionary access:** `dictionary at in field access`

## DB-backed data
Typst does **not** read `.db` directly. Export DB → CSV/JSON and load with `csv()`/`json()`.

# Typst Style Guide (Generic)

## Tone & Consistency
- Match the local document style or template.
- Keep prose concise; prefer short paragraphs and clear claims.

## Figures, Tables, Labels
- Wrap images/tables in `#figure(...)` with `caption:` and `<label>`.
- Reference with `@label` in text; citations with `@bib_key`.
- Keep captions short and consistent across sections.

## Layout Patterns
- Use `#grid(...)` for multi-panel figures.
- Use `#wrap-content(...)` for side-by-side text and figures.

## Data Tables
- Prefer hard-coded tables unless asked for dynamic data.
- If dynamic: load via `csv()` / `json()` and map/flatten.

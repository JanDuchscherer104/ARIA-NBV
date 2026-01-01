# Documentation index update (Quarto)

## What changed

- Updated the main docs landing page navigation list to include **all** Quarto pages under `docs/contents/**`.
- Updated the Quarto sidebar (`docs/_quarto.yml`) so every `.qmd` page is reachable via the docked navigation (plus a `Home` entry).

## Files touched

- `docs/index.qmd`: refreshed the “Documentation Navigation” section (removed duplicates, added missing pages).
- `docs/_quarto.yml`: expanded `website.sidebar.contents` and fixed navbar label typo (`Ressources` → `Resources`).

## Notes / gotchas

- Some literature pages (e.g. `docs/contents/literature/*.qmd`) do not use YAML front matter. Quarto will still infer titles from the first heading, but if you ever want fully consistent sidebar titles/metadata, consider adding minimal YAML (`---\ntitle: ...\n---`) to those pages.
- The docs project uses `output-dir: .` and has `_site/` + `_freeze/` directories; running `quarto render` can update generated files in-place. Keep this in mind if you want “source-only” diffs.


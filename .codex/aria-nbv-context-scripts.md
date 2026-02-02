# aria-nbv-context scripts check (2026-01-26)

## Summary
- Verified and fixed the aria-nbv-context skill scripts so they run from any CWD and match the documented paths.
- Updated skill documentation and internal DB to reference the correct script locations under `.codex/skills/aria-nbv-context/scripts`.

## Changes
- Hardened `.codex/skills/aria-nbv-context/scripts/nbv_get_context.sh`:
  - Auto-detects repo root based on script location.
  - Defaults to venv python at `oracle_rri/.venv/bin/python` when available.
  - Accepts relative/absolute `--root` overrides.
- Hardened `.codex/skills/aria-nbv-context/scripts/nbv_qmd_outline.sh` and `nbv_literature_search.sh`:
  - Resolve repo root automatically.
  - Validate target directories exist.
  - Use `rg` when available, fallback to `grep`.
- Updated `.codex/skills/aria-nbv-context/scripts/nbv_typst_includes.py`:
  - Defaults resolve relative to repo root, even when invoked outside repo.
  - `--paper`/`--slides` now default to `root/paper/main.typ` and `root/slides`.
- Updated `.codex/skills/aria-nbv-context/SKILL.md` to reference the correct script paths.
- Updated `.codex/AGENTS_INTERNAL_DB.md` to note that these helper scripts live under `.codex/skills/...`.

## Verification
- Ran the scripts from repo root and `/tmp`:
  - `.../nbv_qmd_outline.sh` returns doc headings.
  - `.../nbv_literature_search.sh "NBV"` returns matches.
  - `.../nbv_typst_includes.py` prints include graph for the paper.
  - `.../nbv_get_context.sh packages` runs successfully (output redirected to file).

## Follow-ups / Suggestions
- Optional: handle `BrokenPipeError` in `oracle_rri/scripts/get_context.py` to avoid noisy errors when piping to `head`.
- Optional: consider adding repo-root wrapper scripts (e.g., `scripts/nbv_qmd_outline.sh`) if you want the shorter paths in docs.

## Repackaging (2026-01-26)
- Added repo-level wrappers in `scripts/`:
  - `scripts/nbv_get_context.sh`
  - `scripts/nbv_qmd_outline.sh`
  - `scripts/nbv_literature_search.sh`
  - `scripts/nbv_typst_includes.py`
- Updated `.codex/skills/aria-nbv-context/SKILL.md` and `.codex/AGENTS_INTERNAL_DB.md` to mention wrappers.

## Skill repackaging (2026-01-26)
- Added `.codex/codex_make_context.md` structure + cheat-sheet `rg` queries to the skill's SKILL.md.

## Outline graphs update (2026-01-26)
- Added `nbv_qmd_outline.py` to emit nested section outlines for QMD files.
- Updated `nbv_qmd_outline.sh` to call the Python outline script.
- Updated `nbv_typst_includes.py` to emit include + heading graphs (mode=graph) and keep includes-only mode.

## Mermaid UML filtering (2026-01-26)
- Added `CONTEXT_MERMAID_EXCLUDE` in Makefile (default: data.downloader).
- `make context` uses `scripts/filter_mermaid.py` to remove excluded namespaces from the UML diagram printed to stdout and stored in `.codex/codex_make_context.md`.

## Mermaid UML defaults (2026-01-26)
- Default stdout UML excludes: data.downloader, vin.experimental, app.
- Snapshot file `.codex/codex_make_context.md` stores unfiltered UML.

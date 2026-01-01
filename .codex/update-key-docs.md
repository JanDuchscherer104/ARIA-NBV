# Task: Update AGENTS Key Documentation Files (2025-11-22)

## Notes
- Ran `make help`, `conda run -n aria-nbv make context`, and `conda run -n aria-nbv make context-dir-tree` to refresh project snapshots.
- Reviewed `docs/index.qmd` and `docs/contents/todos.qmd` to align the documentation pointers with current navigation and priorities.
- Updated `AGENTS.md` Key Documentation Files section to reflect the latest doc layout (aria_nbv_package, data_pipeline_overview, ext-impl references) and fixed relative paths.

## Suggestions
- Periodically rerun `make context-qmd-tree` when docs change to keep the AGENTS pointers accurate.
- Address the TODOs in `docs/index.qmd` (abstract, resource links) to reduce drift between roadmap and documentation.

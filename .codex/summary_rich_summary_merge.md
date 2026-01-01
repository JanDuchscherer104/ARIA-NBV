# Summary/Rich Summary Merge

## Summary
- Merged `summary.py` into `rich_summary.py` and removed local helper defs from `rich_summary`.
- `rich_summary` now accepts `tree_dict=None`.

## Changes
- `oracle_rri/oracle_rri/utils/rich_summary.py`: added `summarize` and shared helpers, refactored rendering to use module-level functions, made `tree_dict` optional.
- `oracle_rri/oracle_rri/utils/console.py`: import `summarize` from `rich_summary`.
- `oracle_rri/oracle_rri/utils/__init__.py`: export `summarize` from `rich_summary`.
- Removed `oracle_rri/oracle_rri/utils/summary.py`.

## Notes / Suggestions
- If you want type-specific overloads for `summarize`, add them in `rich_summary.py` without reintroducing duplicated logic.

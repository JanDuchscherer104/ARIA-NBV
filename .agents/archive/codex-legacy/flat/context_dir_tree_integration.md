# Context dir-tree integration (2025-11-25)

- Added `_context_dir_tree_print` helper to reuse the oracle_rri tree dump across targets.
- `make context` now appends the directory tree after the class diagram/docstring output, so agents get structure + symbols in one run.
- `make context-dir-tree` delegates to the same helper to keep output consistent and avoid duplicated command blocks.
- Verified by running `make context` and `make context-dir-tree`; both complete successfully and print the tree once.

Potential follow-ups:
- If context output grows too long, consider a `CONTEXT_TREE=0` toggle to skip the tree when unnecessary.

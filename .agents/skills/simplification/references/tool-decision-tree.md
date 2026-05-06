# Simplification Tool Decision Tree

Use this reference after a local `rg` or narrow file read shows that a cleanup
surface is broader than one file.

## Discovery

- Use `rg` for tiny or local checks.
- Use code-index when available for broader symbol-heavy discovery:
  `set_project_path`, `build_deep_index`, `find_files`,
  `search_code_advanced`, `get_file_summary`, and `get_symbol_body`.

## Analysis

- Decide whether a file is worth simplifying with `get_file_summary`.
- Inspect one helper, method, or class with `get_symbol_body`.
- Estimate call-site count with `search_code_advanced`.
- Use package or file analyzers only after code-index narrows the package or
  file.

## Guardrails

- Analyzer output is advisory, not authoritative.
- Repo ownership, active contracts, and behavior-preservation beat generic
  refactoring advice.
- Do not use analyzer suggestions to widen APIs or preserve stale surfaces.

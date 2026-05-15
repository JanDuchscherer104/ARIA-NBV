---
id: 2026-05-15_data_handling_typed_public_imports
date: 2026-05-15
title: "Data Handling Typed Public Imports"
status: done
topics: [data-handling, typing, public-api]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/data_handling/__init__.py
  - aria_nbv/tests/data_handling/public_api_typing_contract.py
---

## Task

Replace lazy `aria_nbv.data_handling` package-root exports with explicit public
re-exports so mypy and IDEs see valid class, alias, constant, and function
types from the canonical data-handling import path.

## Method

Removed `_LAZY_EXPORTS` and `__getattr__` from the package root, then imported
all public names directly while preserving the required `_raw`-first import
order for modules that may indirectly import raw view types from
`aria_nbv.data_handling` during package initialization.

Added `tests/data_handling/public_api_typing_contract.py` as a static mypy
contract for formerly lazy exports. The file is intentionally not collected by
pytest.

## Verification

- `cd aria_nbv && uv run mypy tests/data_handling/public_api_typing_contract.py`
- `cd aria_nbv && uv run pytest tests/data_handling/test_public_api_contract.py`
- `cd aria_nbv && uv run ruff format --check aria_nbv/data_handling/__init__.py tests/data_handling/public_api_typing_contract.py`
- `cd aria_nbv && uv run ruff check aria_nbv/data_handling/__init__.py tests/data_handling/public_api_typing_contract.py`

## Canonical State Impact

No canonical memory update needed. The public data-handling root remains the
canonical import path; it now exposes static types directly instead of through
runtime lazy resolution.

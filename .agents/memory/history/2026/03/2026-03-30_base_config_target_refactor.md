---
id: 2026-03-30_base_config_target_refactor
date: 2026-03-30
title: "BaseConfig target refactor"
status: done
topics: [config, utils, console]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/utils/base_config.py
  - aria_nbv/aria_nbv/utils/console.py
  - aria_nbv/aria_nbv/utils/__init__.py
  - aria_nbv/aria_nbv/utils/grad_norms.py
---

task
- Remove the `Generic[TargetType]` and `NoTarget` pattern from `BaseConfig`, using a nullable `target` surface instead.

method
- Updated `BaseConfig` to default `target` to `None`, return early from `setup_target()` when unset, and keep legacy `BaseConfig[T]` subclass syntax working via `__class_getitem__`.
- Added a small `Console.from_callsite()` compatibility constructor and used it for `BaseConfig` error logging.
- Removed stale `NoTarget` exports/imports and simplified `GradNormLoggingConfig` to inherit plain `BaseConfig`.

verification
- `ruff check aria_nbv/aria_nbv/utils/base_config.py aria_nbv/aria_nbv/utils/console.py aria_nbv/aria_nbv/utils/__init__.py aria_nbv/aria_nbv/utils/grad_norms.py`
- `aria_nbv/.venv/bin/python -m pytest aria_nbv/tests/test_base_config_toml.py`

canonical state impact
- None.

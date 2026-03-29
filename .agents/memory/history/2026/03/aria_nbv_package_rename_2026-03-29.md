---
id: 2026-03-29_aria_nbv_package_rename
date: 2026-03-29
title: "Rename tracked Python workspace and package to aria_nbv"
status: done
topics: [package-layout, packaging, docs, tooling]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/DECISIONS.md
files_touched:
  - aria_nbv/pyproject.toml
  - Makefile
  - pytest.ini
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/DECISIONS.md
---

Task: rename the tracked Python workspace and import package from `oracle_rri` to `aria_nbv`.

Method:
- moved the tracked subtree from `oracle_rri/` to `aria_nbv/`
- renamed the inner package to `aria_nbv/aria_nbv`
- updated packaging metadata, editable install state, Makefile/context tooling, tests, and active docs
- refreshed `aria_nbv/uv.lock` and re-synced the moved venv with `uv sync --all-extras`

Findings:
- the moved venv still pointed at the old editable install until `uv sync` rebuilt the local package metadata
- most remaining `oracle_rri` strings are intentional Oracle RRI concept names, cache directory names, or module filenames such as `oracle_rri_labeler.py`
- the stale `oracle_rri/` directory was only local residue (`.coverage`, `__pycache__`, caches) and was removed after moving `.coverage`

Verification:
- `uv lock` in `/home/jandu/repos/NBV/aria_nbv`
- `uv sync --all-extras` in `/home/jandu/repos/NBV/aria_nbv`
- `aria_nbv/.venv/bin/python` import check for `aria_nbv`, `aria_nbv.lightning.cli`, and `aria_nbv.configs.path_config`
- `aria_nbv/.venv/bin/python -m pytest -s /home/jandu/repos/NBV/aria_nbv/tests/test_pathconfig_isolation_regression.py /home/jandu/repos/NBV/aria_nbv/tests/test_console.py /home/jandu/repos/NBV/aria_nbv/tests/lightning/test_reduce_lr_on_plateau_config.py -q`

Canonical state impact:
- updated project state and decisions to reflect the `aria_nbv/aria_nbv` workspace layout and `aria_nbv/.venv` environment path

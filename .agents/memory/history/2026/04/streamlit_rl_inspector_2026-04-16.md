---
id: 2026-04-16_streamlit_rl_inspector
date: 2026-04-16
title: "Add Streamlit RL inspector"
status: done
topics: [streamlit, rl, app]
confidence: high
canonical_updates_needed: []
---

## Verification

- `/home/jandu/repos/NBV/aria_nbv/.venv/bin/python -m ruff check aria_nbv/aria_nbv/app/app.py aria_nbv/aria_nbv/app/config.py aria_nbv/aria_nbv/app/panels.py aria_nbv/aria_nbv/app/panels/__init__.py aria_nbv/aria_nbv/app/panels/candidates.py aria_nbv/aria_nbv/app/panels/rl.py aria_nbv/tests/app/panels/test_rl_panel.py aria_nbv/tests/app/panels/test_candidates_panel.py`
- `/home/jandu/repos/NBV/aria_nbv/.venv/bin/python -m pytest -s aria_nbv/tests/app/panels/test_rl_panel.py aria_nbv/tests/app/panels/test_candidates_panel.py`

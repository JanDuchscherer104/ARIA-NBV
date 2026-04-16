---
id: 2026-04-16_mojo_benchmark_reporting
date: 2026-04-16
title: "Add Mojo acceleration planning skill and benchmark reporting"
status: done
topics: [mojo, skills, benchmarks, context7]
confidence: high
canonical_updates_needed: []
files_touched:
  - path: .agents/skills/mojo-nbv-acceleration/SKILL.md
    kind: skill
  - path: aria_nbv/aria_nbv/utils/benchmark_plotting.py
    kind: utility
  - path: aria_nbv/scripts/plot_python_vs_mojo_benchmarks.py
    kind: script
---

## Task

Add an ARIA-NBV-specific Mojo planning skill plus utilities for rendering
Python-vs-Mojo benchmark comparisons.

## Verification

- `/home/jandu/repos/NBV/aria_nbv/.venv/bin/python -m ruff check aria_nbv/aria_nbv/utils/benchmark_plotting.py aria_nbv/scripts/plot_python_vs_mojo_benchmarks.py aria_nbv/tests/utils/test_benchmark_plotting.py`
- `/home/jandu/repos/NBV/aria_nbv/.venv/bin/python -m pytest -s aria_nbv/tests/utils/test_benchmark_plotting.py`

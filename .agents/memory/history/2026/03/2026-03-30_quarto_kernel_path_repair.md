---
id: 2026-03-30_quarto_kernel_path_repair
date: 2026-03-30
title: "Repair stale Quarto and Jupyter Python paths"
status: done
topics: [docs, jupyter, quarto, environment]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/GOTCHAS.md
files_touched:
  - .agents/memory/state/GOTCHAS.md
  - docs/contents/ext-impl/prj_aria_tools_impl.qmd
  - docs/contents/resources/agent_scaffold/state/gotchas.qmd
  - external/openpoints_shim/openpoints_shim/build_helpers.py
artifacts:
  - ~/.local/share/jupyter/kernels/python3/kernel.json
  - ~/.local/share/jupyter/kernels/aria_nbv/kernel.json
---

task
- Remove stale references to the deleted `oracle_rri/.venv` path from Quarto/Jupyter execution and user-facing guidance.

method
- Reinstalled the user `python3` kernelspec from `aria_nbv/.venv` and added a named `aria_nbv` kernelspec.
- Updated the Quarto page that explicitly pinned `jupyter: python3` to use `jupyter: aria_nbv`.
- Updated canonical and generated gotcha docs to recommend `uv run --project aria_nbv pytest` or `aria_nbv/.venv/bin/python -m pytest`.
- Updated the OpenPoints shim helper that inferred CUDA from the old venv path.

verification
- `cd docs && quarto check`
- `cd docs && quarto render contents/ext-impl/prj_aria_tools_impl.qmd`
- `rg -n 'oracle_rri/.venv|/home/jandu/repos/NBV/oracle_rri/.venv' /home/jandu/repos/NBV ~/.local/share/jupyter -g '!docs/_freeze/**' -g '!**/.git/**' -S`

canonical state impact
- Updated `GOTCHAS.md` environment guidance to match the renamed `aria_nbv` project layout and correct root-level `uv` usage.

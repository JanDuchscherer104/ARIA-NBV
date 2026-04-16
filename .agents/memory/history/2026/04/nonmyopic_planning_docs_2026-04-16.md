---
id: 2026-04-16_nonmyopic_planning_docs
date: 2026-04-16
title: "Update non-myopic planning docs and memory"
status: done
topics: [docs, memory, rl, planning, hestia]
confidence: medium
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/OPEN_QUESTIONS.md
files_touched:
  - path: docs/contents/ideas.qmd
    kind: docs
  - path: docs/typst/shared/macros.typ
    kind: typst
  - path: docs/references.bib
    kind: bibliography
  - path: .agents/memory/state/PROJECT_STATE.md
    kind: memory
---

## Task

Update the repo's planning direction and advisor-facing narrative around
non-myopic RRI, discrete-shell RL, Hestia-style hierarchy, and semantic-global
planning extensions.

## Verification

- `/home/jandu/repos/NBV/aria_nbv/.venv/bin/python scripts/validate_agent_memory.py`
- `cd docs && typst compile typst/paper/main.typ --root .` (blocked by an existing PDF image-format issue during import, after fixing the new `dot.op` notation mismatch)

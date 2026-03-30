---
id: 2026-03-30_scaffold_guidance_tightening
date: 2026-03-30
title: "Scaffold Guidance Tightening"
status: done
topics: [scaffold, guidance, agent-memory]
confidence: high
canonical_updates_needed: []
files_touched:
  - AGENTS.md
  - aria_nbv/AGENTS.md
  - aria_nbv/aria_nbv/data_handling/AGENTS.md
  - .agents/references/agent_memory_templates.md
  - docs/contents/resources/agent_scaffold/instructions/repo_guidance.qmd
  - docs/contents/resources/agent_scaffold/instructions/package_guidance.qmd
  - docs/contents/resources/agent_scaffold/instructions/data_handling_boundary.qmd
  - docs/contents/resources/agent_scaffold/references/agent_memory_templates.qmd
---

Task
- Add minimal scaffold guidance based on recent `data_handling` and migration cleanup work.

Method
- Tightened the canonical repo/package/data-handling guidance with a few narrow rules around scoped staging in dirty worktrees, dedicated compatibility wrappers, strict format decisions for on-disk contracts, and `--help` smoke coverage for operator CLIs.
- Added a brief note to the memory template reference about recording staged scope and compatibility decisions when those details matter.
- Regenerated the published scaffold pages from the canonical sources.

Verification
- `./scripts/quarto_generate_agent_docs.py`
- `make check-agent-memory`

Canonical State Impact
- None. This updated guidance and scaffold references, not project truth.

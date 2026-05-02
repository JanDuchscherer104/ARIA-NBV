---
id: 2026-05-02_litkg_agent_memory_backlog
date: 2026-05-02
title: "litkg Agent-Memory Backlog Implementation"
status: done
topics: [litkg, agents-db, memory, skills, kg]
confidence: high
canonical_updates_needed: []
files_touched:
  - AGENTS.md
  - Makefile
  - .configs/litkg.toml
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/refactors.toml
  - .agents/kg/README.md
  - .agents/skills/aria-litkg-memory/SKILL.md
  - .agents/skills/semantic-scholar-litkg/SKILL.md
---

## Task

Implemented the extracted litkg agent-memory backlog: make litkg the default
agent memory router, persist the missing issues/todos with source references,
and add thin repo surfaces for retrieval and refresh.

## Outputs

- Added the `aria-litkg-memory` skill and routed broad memory retrieval,
  claim-check, source-backed routing, and consolidation work to it from
  `AGENTS.md`.
- Added Make wrappers for `kg-capabilities`, `kg-search`, `kg-query`,
  `kg-brief`, `kg-route`, `kg-claim-check`, `kg-related`, `kg-show-paper`, and
  staged refresh targets.
- Expanded `.configs/litkg.toml` with agent state, active backlog, history,
  skills/references, generated context, and experiment evidence source classes.
- Clarified litkg/graphify/Neo4j/Graphiti/MemPalace roles in the KG README,
  semantic-scholar-litkg skill, and config comments.
- Amended `issue-023`, `issue-025`, `todo-018`, `todo-039`, and
  `refactor-014`; added `todo-040` through `todo-045` for the missing vertical
  slices.
- Fixed stale Typst source paths in the litkg profile so capabilities now report
  the Typst source as ready.

## Verification

- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `python3 /home/jd/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/aria-litkg-memory`
- `make check-agent-memory`
- `make kg-capabilities KG_FORMAT=json`
- `make kg-search KG_QUERY='entity-aware RRI' KG_LIMIT=5`
- `make kg-query KG_QUERY='current RRI contract' KG_FORMAT=text`

## Canonical State Impact

No durable state file update is required. This changed agent workflow,
backlog/config/docs surfaces, and left future litkg feature implementation as
source-backed active TODOs.

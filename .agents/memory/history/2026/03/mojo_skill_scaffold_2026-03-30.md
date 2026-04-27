---
id: 2026-03-30_mojo_skill_scaffold
date: 2026-03-30
title: "Added Mojo acceleration skill and Context7 routing"
status: done
topics: [skills, mojo, context7, acceleration]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/references/context7_library_ids.md
  - .agents/skills/mojo-nbv-acceleration/SKILL.md
  - .agents/skills/mojo-nbv-acceleration/references/mojo-context7-summary.md
  - .agents/memory/history/2026/03/mojo_skill_scaffold_2026-03-30.md
---

## Task

Add official Mojo Context7 routing to the repo and create a project-specific skill for deciding how Mojo should be used in `aria_nbv`.

## Method

- Resolved the official Context7 library id for Mojo.
- Queried the official docs for Python interop, Python-importable modules, FFI, and GPU programming.
- Mapped those capabilities onto the current Aria-NBV hot paths and interface constraints.
- Added a new repo skill plus a compact reference note.

## Findings Or Outputs

- The most plausible incremental adoption path is Python-orchestrated Mojo behind narrow helper boundaries.
- The best first repo targets remain geometry-heavy helper paths rather than broad model or training rewrites.
- The new skill captures both the official Mojo surfaces and repo-specific decision rules for when not to port code.

## Verification

- `make check-agent-memory`

## Canonical State Impact

- No canonical state docs changed.

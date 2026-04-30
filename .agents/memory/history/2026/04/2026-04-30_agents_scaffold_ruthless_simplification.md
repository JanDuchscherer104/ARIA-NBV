---
id: 2026-04-30_agents_scaffold_ruthless_simplification
date: 2026-04-30
title: "Agents Scaffold Ruthless Simplification"
status: done
topics: [scaffold, agents, simplification, kg]
confidence: high
canonical_updates_needed: []
files_touched:
  - AGENTS.md
  - .gitignore
  - .agents/kg/README.md
  - .agents/references/human_owner_intent.md
  - .agents/kg/generated/
  - .agents/work/research-and-cleanup/
---

## Task

Applied ruthless simplification to the agent scaffold with plan-grilled removal
decisions for generated KG output and scratch research-cleanup transcripts.

## Method

Kept the canonical scaffold surfaces: root and nested guidance, skills,
references, state memory, backlog TOML, KG config, and KG commands. Removed
tracked generated KG bundles and transcript scratch material because they were
rebuildable or superseded by canonical memory/backlog records. Tightened ignore
rules so future generated KG, generated agent mirrors, and `.agents/work/`
scratch output stay local by default.

## Outputs

Deleted the tracked `.agents/kg/generated/` tree and
`.agents/work/research-and-cleanup/` transcript bundle, removing 122 tracked
files and about 28.7k lines from the scaffold. Removed stale commented guidance
from root `AGENTS.md`, clarified `.agents/kg/README.md` so KG output is local
generated output rather than committed scaffold state, and updated the human
owner intent table to route generated context to ignored generated paths.

## Verification

Ran `make check-agent-memory`, `make agents-db AGENTS_ARGS='validate'`, and
`git check-ignore` for `.agents/kg/generated/`, `.agents/work/`, and
`.agents/generated/`.

## Canonical State Impact

No project-state update is needed. The durable rule is now captured in
`.gitignore`, `.agents/kg/README.md`, and
`.agents/references/human_owner_intent.md`: generated scaffold/KG payloads are
local outputs unless explicitly promoted as curated artifacts.

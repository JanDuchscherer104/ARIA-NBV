---
name: aria-litkg-memory
description: Use when answering broad ARIA-NBV project questions, planning cross-surface work, debugging multi-surface issues, checking current truth, retrieving related code/docs/papers/backlog state, or consolidating agent debriefs through the litkg agent-memory layer.
---

# ARIA litkg Memory

Use this skill when litkg should act as the project memory router, not when you
are changing litkg-rs internals. Use `semantic-scholar-litkg` for toolkit,
TOML-ingestion, Semantic Scholar, backend, or code-graph implementation work.

## Protocol

1. Start at repo root and read `AGENTS.md` if the current task did not already
   provide it.
2. Check backend/source readiness when the task depends on KG freshness:
   `make kg-capabilities` or `make kg-capabilities KG_FORMAT=json`.
3. For broad questions, run `make kg-query KG_QUERY="<question>"`.
4. For task routing, run `make kg-route KG_TASK="<task>"`.
5. For quick retrieval, run `make kg-search KG_QUERY="<terms>"`.
6. Inspect cited canonical sources before treating a retrieved statement as
   current truth.
7. Prefer canonical state, code/tests/configs, and active backlog over debriefs,
   generated context, archives, or external papers when sources conflict.
8. After non-trivial work, update durable memory/backlog when truth changed,
   write the required debrief, and run `make check-agent-memory`.

## Source Authority

Treat source tiers as a ranking contract until litkg retrieval exposes explicit
authority scores:

1. Code, tests, and current configs.
2. `.agents/memory/state/PROJECT_STATE.md`, `DECISIONS.md`,
   `OPEN_QUESTIONS.md`, and `GOTCHAS.md`.
3. Root and nested `AGENTS.md` plus `.agents/skills/**/*.md`.
4. Active `.agents/issues.toml`, `.agents/todos.toml`,
   `.agents/refactors.toml`, and `.agents/resolved.toml`.
5. Current Typst/Quarto authored docs.
6. Generated context artifacts.
7. Debrief history, work notes, and archived scratchpads.
8. External papers and web docs, unless the question is specifically about the
   literature or an external API.

## Common Commands

```bash
make kg-query KG_QUERY="What is the current entity-aware RRI plan?"
make kg-brief KG_TOPIC="VIN offline-store diagnostics"
make kg-route KG_TASK="debug candidate pose frame mismatch"
make kg-search KG_QUERY="CW90 candidate pose"
make kg-related KG_RELATED_PATH="aria_nbv/aria_nbv/rri_metrics/oracle_rri.py"
make kg-claim-check KG_CLAIM="ARIA-NBV is an end-to-end RL policy"
```

Use `KG_FORMAT=json` when another tool or agent needs machine-readable output.

## Consolidation

When a debrief, experiment report, or changed code/docs surface should update
durable truth, retrieve related context first, then propose explicit edits to:

- `.agents/memory/state/PROJECT_STATE.md`
- `.agents/memory/state/DECISIONS.md`
- `.agents/memory/state/OPEN_QUESTIONS.md`
- `.agents/memory/state/GOTCHAS.md`
- `.agents/issues.toml`, `.agents/todos.toml`, or `.agents/refactors.toml`

Do not silently promote episodic notes into canonical memory. Use patch-like
proposals or direct edits only when the user asked for implementation and the
source evidence is clear.

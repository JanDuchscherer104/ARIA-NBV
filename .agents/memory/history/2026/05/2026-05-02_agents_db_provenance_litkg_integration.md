---
id: 2026-05-02_agents_db_provenance_litkg_integration
date: 2026-05-02
title: "Agents DB Provenance And litkg Integration"
status: done
topics: [agents-db, litkg, semantic-scholar, backlog, provenance]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/AGENTS_INTERNAL_DB.md
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/skills/agents-db/SKILL.md
  - .agents/skills/semantic-scholar-litkg/SKILL.md
  - scripts/agents_db.py
  - .agents/external/litkg-rs/crates/litkg-core/src/context_pack.rs
  - .agents/external/litkg-rs/crates/litkg-cli/src/main.rs
  - .agents/external/litkg-rs/crates/litkg-cli/tests/inspect_cli.rs
---

## Task

The user required persisted agents-db issues and todos to include richer context
and pointers to internal sources, papers, external docs, or API docs, and asked
how to improve integration between `agents-db`, `semantic-scholar-litkg`, and
`.agents/external/litkg-rs`.

## Method

Used the `agents-db` and `semantic-scholar-litkg` skills, read the root guidance,
project state, thesis questions, litkg-rs architecture docs, integration spec,
and current active backlog. Ran a litkg context pack and found a concrete adapter
gap: ARIA-NBV uses singular `[[issue]]` / `[[todo]]` tables with `description`
fields, while litkg-rs context-pack parsing expected plural tables and
`summary`.

## Outputs

- Added a provenance contract to `scripts/agents_db.py`: active issues now
  require `context` and `references`; active todos now require `references`;
  `references` must be non-empty and use structured prefixes such as `repo:`,
  `bib:`, `url:`, `s2:`, `context7:`, or `litkg:`.
- Added context and references to all active issues and references to all active
  todos.
- Added `issue-025` and `todo-039` for the remaining deeper integration: typed
  litkg/KG resolution of agents-db source references.
- Updated the agents-db and semantic-scholar-litkg skills with the provenance
  and integration rules.
- Updated litkg-rs context-pack parsing to accept singular/plural backlog tables,
  `description` or `summary`, and to carry `context` and `references` in JSON
  and text action packs.

## Verification

- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `cd aria_nbv && uv run ruff format ../scripts/agents_db.py && uv run ruff check ../scripts/agents_db.py`
- `python3 /home/jd/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/agents-db`
- `python3 /home/jd/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/semantic-scholar-litkg`
- `cargo test --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-core -p litkg-cli context_pack`
- `cargo run --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-cli -- context-pack --config .configs/litkg.toml --repo-root . --task "agents db provenance smoke" --profile thesis-coding --format json`
- `make check-agent-memory`

The provenance smoke confirmed 20 active issues and 29 active todos in the
context-pack output, with references present.

## Canonical State Impact

No additional canonical state update is needed. `.agents/AGENTS_INTERNAL_DB.md`
now records the durable agents-db provenance convention.

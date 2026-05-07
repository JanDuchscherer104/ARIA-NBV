---
id: 2026-05-06_parallel_scaffold_litkg_audit_followup
date: 2026-05-06
title: "Parallel Scaffold litkg Audit Follow-Up"
status: done
topics: [agents-db, litkg, scaffold, skills, memory]
confidence: high
canonical_updates_needed: []
files_touched:
  - AGENTS.md
  - .configs/litkg.toml
  - .agents/references/litkg_quick_reference.md
  - .agents/references/skill_style_guide.md
  - .agents/skills/*/SKILL.md
  - .agents/external/litkg-rs/crates/litkg-core/src/context_pack.rs
  - .agents/external/litkg-rs/crates/litkg-core/src/config.rs
  - .agents/external/litkg-rs/crates/litkg-cli/src/main.rs
  - .agents/external/litkg-rs/crates/litkg-cli/tests/inspect_cli.rs
  - .agents/external/litkg-rs/crates/litkg-graphify/src/lib.rs
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/resolved.toml
  - .agents/memory/history/2026/05/2026-05-06_worker_c_litkg_context_pack_db_capture.md
assumptions:
  - "Scope was limited to scaffold, docs, memory, litkg-rs context-pack, and KG config surfaces."
  - "Existing dirty edits outside the requested record changes were preserved."
---

## Task

Implemented the parallel scaffold and litkg audit follow-up with split worker
slices and supervisor review.

## Method

litkg-rs context packs now expose the documented additive agent-facing fields:
`verb`, `assumptions`, `top_sources`, `required_reads`,
`suggested_next_action`, and `missing_context`, while preserving legacy fields
including `action_plan`, `evidence_spans`, `missing_leaves`,
`missing_context_leaves`, `active_issues`, and `active_todos`. Backlog items
also carry `acceptance` and `verification` when present.

Updated `.configs/litkg.toml` so current thesis roadmap/questions, canonical
memory, thesis proposal, and implementation code rank above historical seminar
evidence. Removed ignored work artifacts from litkg indexed history.

Tightened root and broad skill lanes: `agent-behavior` owns generic
request-traceable work discipline; local discovery, KG retrieval, KG
implementation, docs curation, diagnosis, review, simplification, planning, and
agents-db have clearer metadata handoffs and less duplicated routing prose.

Amended existing active agent DB records without creating a new scaffold epic or
resolving active records. `issue-012` owns research-skill and scaffold-routing
coverage, while future litkg metadata consumption stays routed to `issue-023`,
`issue-025`, and `todo-056`. `todo-056` now captures remaining litkg
context-pack implementation debt and warning-level stale-reference diagnostics.

Also normalized the copyable stale skill-validator command in resolved
todo-040 from a host-specific `/home/jd/.codex` path to
`${CODEX_HOME:-$HOME/.codex}`.

## Verification

- `cd .agents/external/litkg-rs && cargo fmt --all --check` passed.
- `cd .agents/external/litkg-rs && cargo test -p litkg-core context_pack` passed.
- `cd .agents/external/litkg-rs && cargo test -p litkg-cli context_pack` passed.
- `cd .agents/external/litkg-rs && cargo test` passed.
- `make kg-capabilities KG_FORMAT=json` passed.
- `make kg-route KG_TASK="scaffold litkg audit follow-up" KG_FORMAT=json` passed
  and emitted the new context-pack fields plus legacy arrays.
- Repo-local skill validation passed for all `.agents/skills/*`.
- `make agents-db AGENTS_ARGS='validate'` passed.
- `make agents-db` passed.
- `make check-agent-memory` passed.
- Targeted stale-text checks found no ignored work-artifact litkg indexing, no
  stale graphify skill escalation, and no copyable host-specific quick-validator
  active guidance path.

## Canonical State Impact

No canonical memory state update is needed. Durable open work remains in
`issue-012`, `issue-023`, `issue-025`, and `todo-056`.

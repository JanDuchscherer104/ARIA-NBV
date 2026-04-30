---
id: 2026-04-30_litkg_m0_backbone_targets
date: 2026-04-30
title: "litkg-rs M0 Backbone Targets"
status: done
topics: [litkg-rs, agents-db, scaffold, context-pack, schema]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/external/litkg-rs/.agents/issues.toml
  - .agents/external/litkg-rs/.agents/todos.toml
  - .agents/external/litkg-rs/.agents/resolved.toml
  - .agents/external/litkg-rs/.agents/scripts/agents_db.py
  - .agents/external/litkg-rs/.agents/scripts/check_backlog.py
  - .agents/external/litkg-rs/.agents/scripts/check_skills.py
  - .agents/external/litkg-rs/.agents/scripts/check_scaffold.py
  - .agents/external/litkg-rs/crates/litkg-core/src/context_pack.rs
  - .agents/external/litkg-rs/crates/litkg-core/src/schema/
  - .agents/external/litkg-rs/crates/litkg-cli/src/main.rs
  - .agents/external/litkg-rs/crates/litkg-cli/tests/inspect_cli.rs
---

## Task

Implemented the highest-priority litkg-rs M0 backbone targets before advanced
orchestration work: agents-db consistency validation, milestone discipline,
minimum schema/provenance hardening, and the context-pack CLI.

## Method

Updated the litkg-rs agents DB so Auto Research and PR fan-out remain open but
are deferred out of M0. Added hard backlog validation and the `validate` agents
DB alias, then added skills and scaffold validators as deterministic read-only
checks. Extended the core schema ontology and validation helpers, fixed DOI
title-conflict resolver behavior, and added `litkg context-pack` with JSON and
text output for the `agents-scaffold` profile.

## Outputs

Closed the completed M0 records in litkg-rs: `ISSUE-0042`, `ISSUE-0048`,
`TODO-0025`, `TODO-0030`, `TODO-0031`, `TODO-0032`, `TODO-0034`, `TODO-0036`,
`TODO-0038`, and `TODO-0046`. The litkg-rs `ci` target now includes agents DB,
skills, and scaffold checks, and the README documents the context-pack CLI.

## Verification

All planned litkg-rs checks passed from `.agents/external/litkg-rs`:
`make agents-db-check`, `make agents-db AGENTS_ARGS='validate'`,
`make agents-db`, `make skills-check`, `make scaffold-check`,
`cargo fmt --all --check`, `cargo test --all-features`, and
`cargo clippy --all-targets --all-features -- -D warnings`.

## Canonical State Impact

No ARIA-NBV canonical state files need updates. The durable ARIA-facing record is
this debrief; implementation state lives in the nested litkg-rs checkout.

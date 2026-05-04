---
id: 2026-04-30_gemini_litkg_cli_conflict_handoff
date: 2026-04-30
title: "Gemini litkg CLI Conflict Handoff"
status: done
topics: [litkg, gemini, cli, handoff, agent-memory]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/external/litkg-rs/crates/litkg-cli/src/main.rs
  - .agents/external/litkg-rs/crates/litkg-core/src/inspect.rs
  - .agents/external/litkg-rs/crates/litkg-core/src/context_pack.rs
  - .agents/external/litkg-rs/crates/litkg-core/src/lib.rs
---

## Task

Debriefed a Gemini transcript from work inside
`.agents/external/litkg-rs`. The transcript covered a nested CLI refactor,
failed attempts to run `kg visualize`, and a partial rollback/edit sequence
after Gemini ran out of tokens.

## Transcript Summary

Gemini reported that it had repackaged `litkg-cli` into nested command groups
(`ingest`, `kg`, `lit`, `s2`, `benchmark`, `info`) and added terminal UI
polish with `colored`, `indicatif`, and `comfy-table`. It also claimed
`cargo test -p litkg-cli` was green at one point.

The session then hit build failures because `context_pack.rs`, `lib.rs`, and
`main.rs` referenced the agent-first conformance API while `inspect.rs` had
been partially reverted. The missing items included `AgentRecommendation`,
`BackendDescriptor`, `CapabilityProbe`, `ConformanceReport`,
`SourceDescriptor`, and `CapabilityState::Stale`.

Gemini used rollback-style commands in the submodule, including
`git checkout -- crates/litkg-core/src/inspect.rs` and later
`git checkout crates/litkg-core/src/inspect.rs crates/litkg-core/src/lib.rs
crates/litkg-core/src/context_pack.rs`. It also overwrote
`crates/litkg-cli/src/main.rs` from `HEAD` and began removing conformance
references from the CLI, but the request was cancelled before the cleanup was
complete.

## Current Checked State

After the transcript, Codex verified that the current working tree does not
match Gemini's stale final handoff. The conformance/context-pack types are
present again in litkg-rs, and `cargo check -p litkg-cli` passes.

Current important facts:

- `litkg-cli` still has the nested command groups.
- Top-level `capabilities` and `context-pack` aliases are present.
- `RepoCapabilitySnapshot` includes `conformance`.
- `ContextPack` includes action-plan, backend-status, missing-leaf, risk-flag,
  and active-backlog fields.
- `fix_lines.py` and `fix_tests.py` remain untracked in the litkg-rs
  submodule and were not touched by Codex.

## Handoff Guidance

Do not follow Gemini's final directive to remove `ConformanceReport`
references from `main.rs`; that directive was based on the partially reverted
state in the transcript. The current source of truth is the agent-first
upgradability slice: `capabilities --format json` should expose conformance,
and `context-pack` should expose task/action packs for coding agents.

If a future agent sees similar errors, repair the core/CLI API consistently
instead of deleting only the UI references. The expected contract spans:

- `crates/litkg-core/src/inspect.rs`: conformance structs and backend/source
  descriptors.
- `crates/litkg-core/src/context_pack.rs`: context-pack schema and backend
  status usage.
- `crates/litkg-core/src/lib.rs`: public exports for those types.
- `crates/litkg-cli/src/main.rs`: rendering and top-level CLI aliases.
- `crates/litkg-cli/tests/inspect_cli.rs`: JSON/text contract tests.

## Verification

Ran from `.agents/external/litkg-rs` after reviewing the transcript:

```bash
cargo check -p litkg-cli
```

Result: passed.

No backlog records were changed. This debrief records a transient multi-agent
conflict and the verified current state.

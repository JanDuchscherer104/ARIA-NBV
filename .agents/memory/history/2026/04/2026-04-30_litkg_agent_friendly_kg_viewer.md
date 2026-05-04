---
id: 2026-04-30_litkg_agent_friendly_kg_viewer
date: 2026-04-30
title: "litkg Agent-Friendly KG Viewer Filters"
status: done
topics: [litkg, kg, viewer, codex, generated-context]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - .agents/external/litkg-rs/crates/litkg-viewer/src/query.rs
  - .agents/external/litkg-rs/crates/litkg-viewer/src/lib.rs
  - .agents/external/litkg-rs/crates/litkg-cli/src/main.rs
  - .agents/external/litkg-rs/crates/litkg-neo4j/src/lib.rs
  - .agents/memory/state/PROJECT_STATE.md
artifacts:
  - .agents/kg/generated/neo4j-export/nodes.jsonl
  - .agents/kg/generated/neo4j-export/edges.jsonl
---

## Task

Make the litkg native graph useful for Codex/Gemini navigation by allowing
modality-filtered visualization, CLI graph entry search, repo-backed `rg`
search, and descriptive node metadata.

## Change

litkg-rs now has a shared graph query/filter layer for `code`, `docs`,
`generated-context`, `literature`, `memory`, `backlog`, and `external-docs`.
`litkg kg find` exposes the same filtering and search ranking used by the
viewer, including read-only `rg --json` mapping from repo hits to graph nodes.
`litkg kg visualize` accepts modality filters plus entry/focus options and the
viewer exposes quick modality controls, repo `rg`, descriptions, and focus
navigation.

The Neo4j export now includes generated-context nodes from
`docs/_generated/context/source_index.md`, `literature_index.md`,
`data_contracts.md`, and `glossary.jsonl`, so `make context` outputs become
searchable graph surfaces.

## Verification

- `cargo fmt --all --check`
- `cargo test -p litkg-cli -p litkg-viewer -p litkg-neo4j`
- `cargo test --all-features`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `make agents-db-check`
- `cargo run --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-cli -- kg export --config .configs/litkg.toml --format json`
- `cargo run --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-cli -- kg find --config .configs/litkg.toml VinPrediction --modality code --format json`

The regenerated ARIA bundle contains 141 `CodeFile`, 140 `CodeModule`, 1428
`CodeSymbol`, 2021 `IMPORTS`, 1951 `CALLS`, 129 `GeneratedContext`, 88
`DataContract`, and 38 `Concept` labels.

## Canonical State Impact

`PROJECT_STATE.md` now records `kg find` / `kg visualize` as agent-facing KG
entry points and distinguishes generated-context graph nodes from the optional
CodeGraphContext runtime.

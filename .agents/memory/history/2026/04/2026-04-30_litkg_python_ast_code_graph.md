---
id: 2026-04-30_litkg_python_ast_code_graph
date: 2026-04-30
title: "litkg Python AST Code Graph Export"
status: done
topics: [litkg, kg, python, code-graph, neo4j]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - .agents/external/litkg-rs/crates/litkg-core/src/code_graph.rs
  - .agents/external/litkg-rs/crates/litkg-core/src/lib.rs
  - .agents/external/litkg-rs/crates/litkg-core/Cargo.toml
  - .agents/external/litkg-rs/crates/litkg-neo4j/src/lib.rs
  - .agents/memory/state/PROJECT_STATE.md
artifacts:
  - .agents/kg/generated/neo4j-export/nodes.jsonl
  - .agents/kg/generated/neo4j-export/edges.jsonl
---

## Task

The native litkg viewer showed the ARIA-NBV graph as a dense documentation-like
cluster rather than a real Python package graph with files, symbols, imports,
and call relationships.

## Findings

ARIA-NBV already configured `[sources.python]` with
`edges = "codegraphcontext"` and `[backends].code_index = true`, but the native
viewer reads the durable Neo4j JSONL export bundle, not the live
CodeGraphContext runtime. The export path previously emitted coarse
`CodeSurface` nodes without AST-backed `CodeFile`, `CodeModule`, `CodeSymbol`,
`IMPORTS`, or `CALLS` relationships.

## Change

Added a RustPython-based Python AST extractor in litkg-rs and wired it into the
Neo4j export. The export now creates deterministic Python code nodes and
relationships from `.configs/litkg.toml`, preserves unresolved imports as
external code references, and emits call edges only when the target resolves to
a local Python symbol/module.

## Verification

- `cargo fmt --all --check`
- `cargo test -p litkg-core code_graph -- --nocapture`
- `cargo test -p litkg-neo4j ast_backed_python_code_graph -- --nocapture`
- `cargo test -p litkg-neo4j`
- `cargo test -p litkg-cli`
- `cargo test --all-features`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo run --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-cli -- kg export --config .configs/litkg.toml --format json`

The regenerated bundle contains 141 `CodeFile`, 140 `CodeModule`, 1428
`CodeSymbol`, 2021 `IMPORTS`, and 1951 resolved local `CALLS` relationships.

## Canonical State Impact

`PROJECT_STATE.md` now distinguishes the live CodeGraphContext/code-index
runtime from the durable native-viewer export. The export is now AST-backed for
local Python code; CodeGraphContext remains the optional deeper live backend.

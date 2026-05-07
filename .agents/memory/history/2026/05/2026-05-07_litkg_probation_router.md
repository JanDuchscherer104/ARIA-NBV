---
id: 2026-05-07_litkg_probation_router
date: 2026-05-07
status: completed
owner: codex
tags: [litkg, kg, scaffold, memory, routing]
canonical_updates_needed: []
---

# LitKG Probation Router

## Summary

The owner decision is to keep LitKG, but demote it from a default agent brain to
a probationary source-backed router, claim checker, and research-memory layer.
Recent Codex sessions showed LitKG helped with broad thesis framing,
literature/Semantic Scholar work, backlog extraction, and claim checks, while
localized coding and Rerun work still worked better through deterministic
skills, `rg`, and direct file reads.

## Decisions Captured

- Localized one-file coding, narrow test fixes, and obvious file discovery do
  not require LitKG.
- Broad thesis/research/backlog/scaffold/Rerun-contract work may use
  `kg-route`, `kg-query`, `kg-search`, `kg-claim-check`, and `kg-consolidate`.
- Static context-pack and search quality is the next priority; live
  Neo4j/Graphiti/embedding UX remains optional until the static router is
  useful.
- Transcript-derived evidence follows a trust ladder: raw agent messages are
  low trust, raw user messages are higher signal but not canonical, distilled
  user intent plus agent-grounded agreement is a high-trust candidate memory,
  and checked-in memory/backlog/docs/code remain the only current truth.

## Verification

- Pending in this pass: litkg-rs scorer/context-pack tests and KG route smoke
  checks after implementation integration.
- Pending in this pass: `make agents-db AGENTS_ARGS='validate'`, `make
  agents-db`, and `make check-agent-memory`.

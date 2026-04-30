---
id: 2026-04-29_litkg_semantic_scholar_rest_integration
date: 2026-04-29
title: "litkg-rs Semantic Scholar REST Integration"
status: done
topics: [litkg, semantic-scholar, knowledge-graph, literature]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/external/litkg-rs/crates/litkg-core/src/semantic_scholar.rs
  - .agents/external/litkg-rs/crates/litkg-cli/src/main.rs
  - .agents/external/litkg-rs/crates/litkg-neo4j/src/lib.rs
  - .agents/external/litkg-rs/README.md
  - .agents/skills/semantic-scholar-litkg/SKILL.md
---

## Task

Implement Semantic Scholar natively in litkg-rs with minimal local code and no required MCP dependency.

## Method

Added a blocking Rust REST client over `ureq`, using official Semantic Scholar Academic Graph and Recommendations endpoints. The client reads `SEMANTIC_SCHOLAR_API_KEY` from the environment, passes it as `x-api-key`, throttles by default, retries 429/5xx, and supports batch paper lookup, bulk search, single paper lookup, recommendations, and author batch lookup.

## Outputs

The litkg-rs CLI now exposes `enrich-semantic-scholar`, `semantic-scholar-search`, `semantic-scholar-paper`, and `semantic-scholar-recommend`. Registry records can store compact Semantic Scholar metadata, and Graphify/Neo4j outputs surface paper IDs, authors, fields, external IDs, and citation-count properties when available.

## Verification

Ran `cargo fmt --all --check`, `cargo test`, `make agents-db`, and CLI help smoke checks for `semantic-scholar-search` and `enrich-semantic-scholar`.

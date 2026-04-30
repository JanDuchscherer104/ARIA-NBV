---
id: 2026-04-29_aria_litkg_profile_initialization
date: 2026-04-29
title: "ARIA-NBV litkg Profile Initialization"
status: done
topics: [litkg, knowledge-graph, code-index, semantic-scholar]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - .configs/litkg.toml
  - .cgcignore
  - .agents/kg/README.md
  - scripts/kg/index_code.sh
  - scripts/kg/ingest_docs.sh
  - scripts/kg/ingest_papers.sh
  - Makefile
  - AGENTS.md
  - .agents/external/litkg-rs/crates/litkg-core/src/model.rs
  - .agents/external/litkg-rs/scripts/kg/index_code.sh
---

## Task

Initialize an ARIA-NBV knowledge-graph profile that selects the repo sources, declares the representation policy, and wires the existing litkg-rs submodule commands instead of leaving local `scripts/kg/*` stubs.

## Method

Replaced `.configs/litkg.toml` with the runnable litkg-rs config plus ARIA-specific source-selection tables. The profile includes agent guidance/memory, Python source, Quarto docs, Typst docs, local literature assets, Context7 library ids, and optional MarkItDown URL ingestion disabled by default.

Selected graphify-style Markdown/JSON as the primary durable representation, Neo4j export as the traversal/import bundle, CodeGraphContext as the code-symbol runtime, Graphiti as optional temporal doc ingestion, and mempalace as separate memory mining rather than the structural KG backend.

## Findings

The existing ARIA-NBV KG scripts were intentional no-op stubs. They now call the litkg-rs submodule. CodeGraphContext was not installed initially; the litkg-rs bootstrap now falls back to `uv venv --seed --clear` when system `python3 -m venv` lacks `ensurepip`.

Live Semantic Scholar enrichment exposed official-response shape drift: `externalIds` can include numeric values and list fields can be `null`, so the Rust model now normalizes those cases.

The ARIA-NBV paper-ingestion wrapper now sources ignored local `.env` before invoking litkg-rs, allowing `SEMANTIC_SCHOLAR_API_KEY` to stay out of tracked config while still working through `make kg-semantic-enrich`.

## Verification

Validated config parsing with `cargo run --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-cli -- stats --config .configs/litkg.toml --format json`. The initialized ARIA profile finds 56 papers, 9 parsed local TeX assets, 9 local PDFs, 238 sections, 86 figures, 36 tables, and 412 citations.

Ran `make kg-ingest-papers`, `make kg-semantic-enrich`, `make kg-materialize`, and `make kg-export-neo4j`. Semantic Scholar enriched 24 of 56 registry records. The generated Neo4j export contains 772 nodes and 727 edges. CodeGraphContext indexed `aria_nbv/aria_nbv` into local Neo4j with 1 repository, 141 files, 1,234 functions, 209 classes, and 352 modules; `./scripts/kg/index_code.sh --check` passes with Neo4j running.

Rust verification passed with `cargo fmt --all --check`, `cargo test`, and `make agents-db` inside litkg-rs. ARIA-side validation passed with the skill quick validator, shell syntax checks, and `make check-agent-memory`.

After adding the local API key to ignored `.env`, reran `make kg-semantic-enrich`, `make kg-materialize`, and `make kg-export-neo4j`; enrichment still reports 24 of 56 records and generated outputs refresh successfully.

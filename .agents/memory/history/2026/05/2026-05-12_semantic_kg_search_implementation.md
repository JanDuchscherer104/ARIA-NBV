---
id: 2026-05-12_semantic_kg_search_implementation
date: 2026-05-12
title: "semantic kg-search implementation"
status: done
topics: [litkg, kg-search, neo4j, embeddings, bm25, semantic]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/external/litkg-rs/
  - .configs/litkg.toml
  - Makefile
  - scripts/kg/compact_search.jq
  - .agents/todos.toml
  - .agents/resolved.toml
  - .agents/memory/state/PROJECT_STATE.md
---

## Task
Continue Claude's litkg-rs semantic `kg-search` work and implement the
accepted plan covering `todo-066`, `todo-067`, and `todo-070`.

## Method
1. Implemented lexical search in `litkg-viewer` with BM25, Porter stemming,
   config-driven synonyms, fuzzy typo fixups, query-fixup reporting, and
   regression tests.
2. Added hybrid `kg find` search in `litkg-cli`: Ollama query embeddings,
   Neo4j vector-index lookup, BM25/cosine blending, `--lexical-only`,
   `--alpha`, explicit `search_mode`, `mode_reason`, and paper-coverage
   fallback handling.
3. Added Neo4j HTTP vector-query support in `litkg-neo4j` and runtime config
   defaults in `litkg-core`.
4. Added `scripts/kg/load_bundle.py`, `kg-load-bundle`, and
   `kg-refresh-semantic` so the JSONL Neo4j export can be loaded into the live
   runtime DB before enrichment.
5. Extended `enrich_embeddings.py` beyond code symbols so Papers,
   PaperSections, Documents, DocSections, ProjectMemory, memory surfaces,
   generated context, transcripts, and backlog records can receive embeddings.
6. Updated `.configs/litkg.toml` with BM25/hybrid weights and ARIA-NBV synonym
   seeds.

## Outputs
- `kg-load-bundle` loaded 9,711 nodes and 15,516 edges into Neo4j using the
  `LitkgNode` marker label and indexed `litkg_id` matches.
- `KG_SKIP_TOKEN_FILTER=1 KG_EMBED_BATCH_SIZE=16 make kg-enrich` embedded
  6,306/6,306 selected runtime nodes with `qwen3-embedding:4b`.
- Embedded runtime coverage now includes DocSection, Document, PaperSection,
  Paper, ProjectMemory, backlog, memory, and code-symbol rows.
- `kg-search` JSON now returns a wrapper with `results`, `search_mode`,
  `mode_reason`, and `query_fixups`; compatibility helpers still support bare
  hit arrays where needed.
- Active todos `todo-066`, `todo-067`, and `todo-070` were moved to
  `.agents/resolved.toml` with resolution notes.

## Verification
- `.agents/external/litkg-rs`: `cargo test` passed, including doctests.
- `.agents/external/litkg-rs`: `cargo fmt --all` was run during the patch
  cycle; final format check is part of the closing verification.
- `python3 -m py_compile .agents/external/litkg-rs/scripts/kg/load_bundle.py
  .agents/external/litkg-rs/scripts/kg/enrich_embeddings.py` passed.
- `.agents/external/litkg-rs/scripts/kg/load_bundle.py --help` passed.
- `make -n kg-refresh-semantic` produced the expected `kg-refresh-lit`,
  `kg-load-bundle`, and `kg-enrich` command chain.
- `make kg-search KG_QUERY='hierachical viewpoint' KG_FORMAT=json KG_LIMIT=5`
  ran in hybrid mode, reported `hierachical->hierarchical`, and surfaced
  Hestia/hierarchical planning in the top results.
- `make kg-search KG_QUERY='how do hierarchical NBV methods decompose target
  proposal from pose realization' KG_FORMAT=json KG_LIMIT=20` returned Hestia
  as `paper:hestia:section:1` in the top result set.
- `make kg-search KG_QUERY='NBV view selection' KG_FORMAT=json KG_LIMIT=5`
  returned NBV literature hits including VIN-NBV and PB-NBV papers.
- `NEO4J_HTTP_URL=http://127.0.0.1:1 make kg-search KG_QUERY='RRI'
  KG_FORMAT=json KG_LIMIT=3` fell back to `lexical_only` with
  `mode_reason=neo4j_unreachable`.
- `make check-agent-memory` and `python3 scripts/agents_db.py validate` passed
  before the debrief; they should be rerun after this file and the canonical
  state update.

## Canonical State Impact
`.agents/memory/state/PROJECT_STATE.md` now records the implemented semantic
`kg-search` state, current Neo4j embedding coverage, resolved blocker status,
and remaining issue-025 backlog. No additional canonical state update is
pending.

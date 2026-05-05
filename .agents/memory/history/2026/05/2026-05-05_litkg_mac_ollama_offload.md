---
id: 2026-05-05_litkg_mac_ollama_offload
date: 2026-05-05
title: "litkg Mac Ollama Offload"
status: done
topics: [litkg, kg-runtime, ollama, graphiti, embeddings]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/external/litkg-rs/scripts/kg/ollama_http.py
  - .agents/external/litkg-rs/scripts/kg/ingest_docs.sh
  - .agents/external/litkg-rs/scripts/kg/enrich_embeddings.py
  - .agents/external/litkg-rs/.env.example
  - Makefile
  - .agents/references/operator_quick_reference.md
---

## Task

Implemented Mac-offloaded Ollama support for ARIA-NBV litkg runtime refresh while
keeping sources, Neo4j, Graphiti, CodeGraphContext, and generated KG artifacts on
the Ubuntu workstation.

## Method

Added a repo-independent litkg-rs HTTP preflight/helper for Ollama. The KG doc
ingestion and embedding-enrichment scripts now verify `qwen3-embedding:4b` and
`gemma4:26b` over HTTP instead of requiring a local Ubuntu `ollama` CLI or
silently falling back to another Gemma model. The helper normalizes native and
OpenAI-compatible Ollama URLs so SSH reverse tunnels can be used as the default
model access path.
The `qwen3-embedding:4b` model returns 2560-dimensional vectors in the current
Mac Ollama install, so `EMBEDDING_DIM=2560` is now the default for litkg runtime
preflight, Graphiti ingestion, and Neo4j embedding enrichment.
ARIA-NBV stores those defaults in `.configs/litkg.toml` under
`[runtime.ollama]`, and the root `make kg-ollama-check`, `make kg-ingest-docs`,
and `make kg-enrich` paths read that config directly.
The Neo4j `kg-up` helper now recreates the container while retaining mounted
data so stale in-container APOC config duplication does not leave `litkg-neo4j`
stuck in a restart loop.
The ARIA-NBV Makefile exposes `KG_DOC_PATHS` and `kg-ingest-docs-smoke` so
operators can validate Graphiti ingestion with a small document before launching
the slower full default doc set.

## Outputs

- Added `kg-ollama-check` in litkg-rs and the ARIA-NBV root Makefile.
- Updated litkg-rs defaults to `GRAPHITI_LLM_MODEL=gemma4:26b`,
  `EMBEDDING_MODEL=qwen3-embedding:4b`, `EMBEDDING_DIM=2560`, and
  `SEMAPHORE_LIMIT=1`.
- Added `[runtime.ollama]` to `.configs/litkg.toml` so routine ARIA-NBV KG
  commands do not require repeated shell exports.
- Documented the Mac SSH reverse tunnel and Ubuntu environment exports in the
  operator quick reference.

## Verification

- `make -C .agents/external/litkg-rs kg-smoke`
- `python3 .agents/external/litkg-rs/scripts/kg/ollama_http.py self-test`
- `python3 .agents/external/litkg-rs/scripts/kg/ollama_http.py openai-url --base-url http://127.0.0.1:11434`
- unreachable-endpoint smoke for clear `kg-ollama-check` failure output
- `make kg-ollama-check` with the live Mac SSH reverse tunnel
- `EMBEDDING_DIM=1024 make kg-ollama-check` to confirm TOML config wins over a
  stale exported dimension for the ARIA-NBV make target

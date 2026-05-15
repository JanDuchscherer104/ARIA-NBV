---
id: 2026-05-15_kg_search_modality_wrapper
date: 2026-05-15
title: "kg-search Modality Wrapper"
status: done
topics: [litkg, kg-search, makefile, retrieval]
confidence: high
canonical_updates_needed: []
files_touched:
  - Makefile
  - .agents/references/litkg_quick_reference.md
  - .agents/external/litkg-rs/crates/litkg-cli/src/main.rs
---

## Task

Expose the existing `litkg-cli kg find --modality` filter through `make kg-search` as `KG_MODALITY`, without adding aliases such as `paper`.

## Findings

The Make wrapper did not forward modality filters. The underlying CLI already accepted exact modality values, but hybrid vector search merged unfiltered vector hits after the lexical filtered search, so `--modality literature` could still surface `docs` hits in hybrid mode.

## Changes

`make kg-search` now accepts comma-separated `KG_MODALITY` values and expands them to repeated `--modality` flags for both compact and JSON/full output modes.

`litkg-cli kg find` now applies the graph modality filter to Neo4j vector hits before hybrid blending, so modality filters constrain both lexical and semantic results.

The litkg quick reference now documents `KG_MODALITY="literature,docs"` and the exact literature search example.

## Verification

- `make kg-search KG_QUERY="VIN-NBV RRI" KG_MODALITY="literature" KG_LIMIT=5`
- `make kg-search KG_QUERY="VIN-NBV RRI" KG_MODALITY="literature" KG_FORMAT=json KG_LIMIT=5 | jq -r '.results[].modality'`
- `make kg-search KG_QUERY="rollout zarr" KG_MODALITY="code,docs" KG_FORMAT=json KG_LIMIT=8 | jq -r '.results[].modality'`
- `make kg-search KG_QUERY="VIN-NBV RRI" KG_MODALITY="paper"` failed with the expected clap invalid-value error.
- `cd .agents/external/litkg-rs && cargo fmt -p litkg-cli --check`
- `cd .agents/external/litkg-rs && cargo test`
- `make kg-capabilities KG_FORMAT=json`
- `make check-agent-memory`

## Canonical State Impact

No canonical state update is needed. This is a command wrapper and retrieval filter correctness change.

---
id: 2026-05-15_litkg_route_page_pointer_fix
date: 2026-05-15
title: "litkg Route Page Pointer Fix"
status: done
topics: [litkg, kg-route, context-pack, retrieval]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/external/litkg-rs/crates/litkg-core/src/context_pack.rs
  - scripts/kg/compact_route.jq
---

## Task

Investigate why `make kg-route KG_TASK="enrich counterfactual rollouts offline dataset page"` returned too little useful information and too few pointers.

## Findings

The evidence set already contained useful route material, including the ASE dataset page, dataset and rollout skills, data-handling owner guidance, and rollout implementation files. The compact route output was weak because `top_sources` ranked tests and generic thesis docs above the docs/owner surfaces, and the compact jq filter only exposed three sources plus a single `read_first` pointer.

## Changes

`top_sources()` now detects documentation/page routing tasks and gives priority to docs pages, owner guidance, and directly matching implementation contracts over tests. Lexical scoring also accounts for path term matches, so a query mentioning counterfactual rollouts and offline dataset can surface the owning docs and package guidance even when snippet overlap is sparse.

The compact `kg-route` jq output now keeps five top sources, emits a `required_reads` list with reasons, and includes a small `relevant_symbols` list instead of collapsing the route into one pointer.

## Verification

- `cd .agents/external/litkg-rs && cargo fmt -p litkg-core --check`
- `cd .agents/external/litkg-rs && cargo test -p litkg-core context_pack::tests::`
- `cd .agents/external/litkg-rs && cargo test`
- `make kg-route KG_TASK="enrich counterfactual rollouts offline dataset page"`

The fixed compact route now starts with `docs/contents/ase_dataset.qmd`, `.agents/skills/dataset-cache-ops/SKILL.md`, `.agents/skills/counterfactual-rollout-planner/SKILL.md`, `aria_nbv/aria_nbv/data_handling/AGENTS.md`, and `aria_nbv/AGENTS.md`.

## Canonical State Impact

No canonical state update is needed. This was a routing and retrieval-quality fix for existing context-pack evidence.

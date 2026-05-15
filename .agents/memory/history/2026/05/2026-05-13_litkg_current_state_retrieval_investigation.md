---
id: 2026-05-13_litkg_current_state_retrieval_investigation
date: 2026-05-13
title: "litkg Current State Retrieval Investigation"
status: done
topics: [litkg, kg-search, kg-claim-check, retrieval, grounding]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/GOTCHAS.md
files_touched:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/GOTCHAS.md
  - .agents/memory/history/2026/05/2026-05-13_litkg_current_state_retrieval_investigation.md
---

## Task

Investigate the current state of `.agents/external/litkg-rs` for research and
ML-development grounding, routing, retrieval, and claim-check usefulness.

## Method

Read root `AGENTS.md`, source-order guidance, litkg quick reference,
`semantic-scholar-litkg`, `aria-litkg-memory`, and the litkg-rs owner guide.
Inspected `.configs/litkg.toml`, `todo-062` through `todo-073`, issue-025,
`graph_hit_from_vector`, `top_sources`, `claim_verdict`, and
`calculate_weighted_score`. Reproduced the main KG paths with:

- `make kg-status`
- `make kg-capabilities KG_FORMAT=json`
- `make kg-search KG_QUERY='VIN-NBV RRI' KG_FORMAT=json`
- direct lexical-only `litkg-cli kg find --lexical-only 'VIN-NBV RRI'`
- `make kg-claim-check KG_CLAIM='VIN-NBV introduces Relative Reconstruction Improvement (RRI), an oracle label computed from point-mesh reconstruction-error reduction after adding a query view' KG_FORMAT=json`
- `make kg-route KG_TASK='smoke' KG_FORMAT=json`
- `cd .agents/external/litkg-rs && cargo test`
- `python3 scripts/agents_db.py validate`
- `make check-agent-memory`

## Findings

Runtime health is good: `kg-status` passed, Ollama and Neo4j HTTP were up, and
Neo4j contained embedded `Paper`, `PaperSection`, `DocSection`,
`ProjectMemory`, generated-context, backlog, and code-symbol rows. The old
runtime paper-coverage blocker is resolved.

Lean ContextPack behavior is live: the smoke route was 41,843 bytes by default
and 231,308 bytes with `KG_VERBOSE=1`, matching the intended compact default
versus legacy full payload split.

Paper provenance is partly fixed: all 84 exported `Paper` nodes now carry
non-null `source_path`, and lexical-only search returns VIN-NBV paper sections
with `authority=canonical`. However, hybrid vector hits still hardcode
`authority=semantic` in `graph_hit_from_vector`, so `todo-072` remains real.

The RRI claim-check still fails: the VIN-NBV RRI claim returns
`verdict=unverifiable`, `confidence=0.2`, with no supporting evidence. The
current evidence set contains no `docs/contents/literature/vin_nbv.qmd` spans,
even with a high evidence budget. Inspection points to `configured_source_paths`
sorting and truncating the configured source list before the curated literature
file is reached, plus the existing `top_sources`/freshness issue. `todo-073`
should start by diagnosing source admission, not only top-source freshness.

Search quality is useful but not yet robust. Exact `Hestia` finds the curated
page in hybrid mode, but broad semantic queries can rank adjacent literature or
archived notes above the intended source. `Vinformer` currently returns archived
VIN notes instead of the code symbol, confirming that `todo-071` is still a
real code-identifier retrieval issue.

## Verification

`cargo test` in `.agents/external/litkg-rs` passed across CLI, core, graphify,
Neo4j, viewer, schema, idempotency, and doc-test surfaces. `scripts/agents_db.py
validate` passed. `make check-agent-memory` passed after these memory updates.

## Canonical State Impact

Updated `PROJECT_STATE.md` and `GOTCHAS.md` to stop describing resolved
code-only vector coverage and missing paper-node provenance as current truth.
Remaining immediate blockers are `todo-072` and `todo-073`, followed by
`todo-071` and retrieval dedup/authority cleanup.

---
id: 2026-05-13_litkg_grounding_chain_062_to_074
date: 2026-05-13
title: "litkg grounding + verdict chain 062 -> 074"
status: done
topics: [litkg, kg-search, kg-claim-check, kg-route, verdict, hybrid, mcp, kg-doctor, hooks, ranking, authority]
confidence: high
canonical_updates_needed: []
---

## Task
End-to-end grounding and verdict for ARIA-NBV research / ML-dev claims:
agents should be able to call `make kg-claim-check 'VIN-NBV introduces
RRI'` and get a `supported` verdict backed by literature, with sane
hybrid search authority, lean default payloads, automated refresh, an
MCP escape hatch for typed queries, and a single health-check
command. Five days of work converged today.

## Method (chronological)

1. **`todo-062` — paper provenance.** `crates/litkg-neo4j/src/lib.rs`
   `Neo4jSink::export` now plumbs `ParsedPaper.provenance[0]` into
   `source_path` / `repo_path` for `Paper` and `PaperSection` nodes;
   BibTeX-only papers fall back to `docs/references.bib`. Parent
   `.configs/litkg.toml` tiers `docs/contents/literature/**/*.qmd`,
   `docs/literature/tex-src/**/*.tex`, and `docs/references.bib` at
   1.5 (canonical). All 84 paper nodes now carry non-null source_path.
   Commits: submodule `1b88bc5`, parent `7dcfcea`.

2. **`todo-063` — lean ContextPack default.** `ContextPackRequest`
   gains a `lean: bool` (default true via the `--full` CLI flag).
   `build_context_pack` leaves bulk fields empty when lean; serde
   guards strip them. Trivial-task kg-route JSON drops from 230 KB to
   ~42 KB. Verdict path preserved (verdict/confidence/
   supporting/contradicting_evidence stay populated regardless of
   lean). Makefile `KG_VERBOSE=1` now passes `--full`. Commits:
   submodule `2973de2`, parent `907cc3e`.

3. **Refresh hooks (`dc29f8e`).** Bucket-aware
   `scripts/kg/auto_refresh.sh` dispatches `kg-refresh-light` always,
   `kg-refresh-code` on `aria_nbv/**/*.py` changes, and
   `kg-export-neo4j + kg-load-bundle + kg-enrich` on
   memory/docs changes. Wired into Claude `Stop`, Codex `Stop`, Gemini
   `SessionEnd`, and a tracked git `post-commit` hook
   (`scripts/git_hooks/post-commit`). `make install-hooks` copies
   `.codex/hooks.example.json` to `.codex/hooks.json` (gitignored) and
   symlinks the git hook. Lock-protected, idempotent on no-op.

4. **`todo-072` — hybrid mode honors authority_tiers.** Earlier the
   Codex commit `a62e487` hardcoded `authority="semantic"` for every
   vector-index hit. Resolved 2026-05-13 by threading `&RepoConfig`
   into `apply_hybrid_vector_search` → `blend_hybrid_hits` →
   `graph_hit_from_vector`; the latter now calls
   `calculate_weighted_score(path, cosine, authority_tiers)` when the
   `Neo4jVectorHit.repo_path` is `Some`. `paper:vin-nbv` returns
   `authority="canonical"`, `score_authority=1.5`, `score_freshness=0.91`
   in live hybrid search (was `semantic`/1.0/1.0).

5. **`todo-073` — canonical literature reaches `top_sources`.**
   Three coordinated fixes:
   - `configured_source_paths` cap of 512 now preserves
     canonical-tier paths first when alphabetical truncation kicks in
     (aria_nbv/**/*.py expanded to 22k+ paths pre-exclude).
   - `truncate_spans_to_budget` reserves
     `CANONICAL_RESERVED_TOKENS=2400` (capped at budget/4) for
     canonical spans before the score-ordered budget filter.
   - `top_sources` does a two-pass admission: pass 1 fills 8 slots
     score-ordered (preserves owning-route precedence test for
     code-fix tasks), pass 2 rescues up to 4 canonical-tier matches
     with term overlap. `apply_confidence_floor` exempts canonical
     entries.
   `docs/contents/literature/vin_nbv.qmd` now reaches `evidence_spans`
   for the RRI claim.

6. **`neo4j-cypher` MCP profile (`ead5240`).** Typed-query escape
   hatch for queries the wrappers can't express (multi-hop joins,
   filtered vector lookups). `make kg-mcp-install` prints the
   gateway-side `mcp-add` / `mcp-config-set` / `mcp-create-profile
   litkg-cypher` commands. Read-only by default; APOC already loaded
   in the repo's Neo4j docker. Schema cheat sheet + 3 example Cypher
   queries documented in `litkg_quick_reference.md`.

7. **`make kg-doctor` (`221530a`).** Nine probes (Ollama tunnel,
   embedding round-trip, Neo4j HTTP, APOC count, vector index state,
   embedding coverage vs bundle, refresh-stamp age, stale lock, kg
   search smoke). Text table or JSON via `KG_DOCTOR_ARGS='--format
   json'`; exits non-zero on red. Wired into the tail of every
   `auto_refresh.sh` dispatch via `--soft --format json` for a
   structured health snapshot in `.agents/kg/.refresh.log`.

8. **`todo-074` — verdict canary closes.** Even after 072 + 073,
   `make kg-claim-check 'VIN-NBV introduces RRI'` returned
   `unverifiable` because `top_sources` admission was dominated by
   test files and thesis prose, AND `classify_claim_span` only
   recognised three hand-coded claim shapes. Two-part fix
   (`abc4510` / `443581c`):
   - `top_sources` adds a literature-class reservation pass (cap +2)
     for paths under `docs/contents/literature/**`,
     `docs/literature/tex-src/**`, or `docs/references.bib`.
   - `classify_claim_span` gains three new `positive_* /
     evidence_*` helper pairs (RRI definition, hierarchical NBV,
     ARIA-NBV objective) slotted above the generic overlap fallback.
   - `deferred_main_simulator_claim` picks up `"only if"` /
     `"thesis-grade"` markers so the Habitat verdict stays
     `unverifiable` under the broader admission.
   New unit test `context_pack_claim_check_handles_diverse_research_shapes`
   covers six claim shapes. The RRI canary now returns
   `verdict=supported` with `confidence=1.0`.

## Findings

- **The verdict pipeline has three orthogonal layers**, and a working
  verdict requires all three to align: data must reach
  `evidence_spans` (path globs + budget + freshness + path cap),
  must survive `top_sources` admission (score-order + canonical
  rescue + literature reservation + confidence floor), and must
  classify as `Supports` in `classify_claim_span` (shape patterns +
  evidence predicates + overlap fallback). Fixing the canary required
  touching all three. Earlier rounds of work only saw one layer at a
  time.
- The `is_search_stopword` + `context_pack_stopword` filters drop
  `"nbv"`, `"aria"`, and a handful of other terms during `task_terms`
  extraction. Claim-side pattern detection should match the **raw
  claim string** (which still contains them) rather than the filtered
  terms set.
- Pattern-based heuristics scale linearly with claim shapes. A
  chat-model fallback (gemma-4 via Ollama) is the path to non-linear
  scaling but adds a tunnel dependency and ~80 ms latency per claim.
  Deferred until the shape catalog gets uncomfortably long.

## Verification

Live regression sweep (2026-05-13, all green):

| Claim | Verdict | Confidence |
|---|---|---|
| RRI canary (VIN-NBV introduces RRI…) | supported | 1.0 |
| V0 GT-OBB sanity | supported | 1.0 |
| V1 actor-visible | contradicted | 0.9 |
| Habitat as main simulator | unverifiable | 0.2 |

`make kg-doctor`: all 9 checks green (no yellows). 107/107 cargo tests
pass.

## Canonical State Impact

- `.agents/memory/state/GOTCHAS.md` updated 2026-05-13 with refreshed
  hybrid-authority + literature-admission + verdict-heuristic +
  `neo4j-cypher` MCP notes.
- `.agents/memory/state/DECISIONS.md` and `PROJECT_STATE.md`
  unchanged — the work fits the existing decisions (semantic
  search uses Neo4j vector, compact-by-default agent output, etc.).
- `.agents/resolved.toml` records 062 / 063 / 067 / 070 / 072 / 073 /
  074 with full resolution notes; `.agents/todos.toml` now carries
  064 (stale lit audit), 068/069 (Graphiti/MemPalace deferred), 071
  (modality-aware stemming), and 075 (refresh-path cleanup, filed
  this session).

## Open Backlog

- **`todo-064`** stale-literature audit — low priority cleanup.
- **`todo-068`** Graphiti integration — deferred, no concrete
  temporal-query workflow yet.
- **`todo-069`** MemPalace audit — deferred, capability/config
  inconsistency.
- **`todo-071`** modality-aware stemming for code-symbol queries —
  medium priority follow-up to the BM25 work; not blocking research.
- **`todo-075`** refresh-path cleanup —
  `graphify_rebuild_command` unset blocks `kg-refresh-lit`;
  `kg-refresh-code` Neo4j-probe disagrees with `auto_refresh.sh`'s
  HTTP probe. Both surface as `warn:` in the hook log but degrade
  silently.

## Confidence

High. The chain is verifiable end-to-end (`make kg-claim-check 'VIN-NBV
introduces RRI'` → supported with literature in top_sources), the
heuristic is regression-tested across six claim shapes, and the doctor
gates partial-state issues. The only remaining "tribal" fragility is
the verdict heuristic's linear scaling with claim shapes; that's a
known limitation tracked in todo-074's resolution note.

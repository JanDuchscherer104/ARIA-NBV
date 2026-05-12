---
id: 2026-05-12_semantic_kg_search_prereqs_5_todos_filed
date: 2026-05-12
title: "semantic kg search prereqs + 5 todos filed"
status: done
topics: [litkg, kg-search, neo4j, embeddings, semantic, signal-noise]
confidence: high
canonical_updates_needed: []
---

## Task
Plan semantic kg-search via the embedding model (per user direction
2026-05-12), ship the small infrastructure pieces that don't depend on
the full Rust work, and file the remaining work as agents-db backlog
with discovery context recorded.

## Method
1. Audited the four configured semantic-search backends (Neo4j +
   APOC, Graphiti, MemPalace, the qwen3-embedding ollama model).
2. Verified end-to-end that the Neo4j vector index
   `kg_embedding_index_2560` already exists (created idempotently at
   `enrich_embeddings.py:502`) and that 1936 nodes carry embeddings.
3. Smoke-tested vector search with a raw Cypher call using the user's
   typo'd query "hierachical viewpoint" - returned 5 code-symbol hits
   with cosine ~0.74. The query path works.
4. **Critical discovery**: live Neo4j contains only CodeGraphContext
   code symbols (1234 Function + 352 Module + 209 Class + 141 File);
   zero Paper, PaperSection, DocSection, or ProjectMemory nodes are
   embedded. The JSONL export has them but the runtime DB doesn't.
   Filed as todo-070 as a prerequisite for Phase B.
5. Updated todo-067 to reflect that B1 (vector-index creation) is
   already done and that the missing prerequisite is paper/doc/memory
   ingestion (todo-070), not index plumbing.
6. Shipped B3 (KG_NEO4J_AUTO_UP opt-in) and B4 (search_mode +
   query_fixups scaffolding) in this session; they work regardless of
   which B0 route (Neo4j-side vs JSONL-sidecar) the user picks.

## Findings
- `enrich_embeddings.py:502 create_vector_index` already creates
  `kg_embedding_index_2560` (HNSW cosine, dim 2560). B1 from the
  earlier plan is therefore done.
- Live Neo4j embedded-node inventory (via Cypher SHOW VECTOR INDEX +
  MATCH count): 1936 KGEmbeddingNodes, all code symbols.
- Manual semantic-query smoke with ollama embeddings + Cypher
  `db.index.vector.queryNodes` works end-to-end (~80 ms query
  embedding + ~10 ms vector search).
- Wiring the query path into `run_kg_find` (todo-067 B2) is still
  needed but will only surface code symbols until todo-070 lands.
- Filed 5 todos against issue-025: todo-066 Phase A (BM25 + stem +
  synonyms + fuzzy lexical floor), todo-067 Phase B (Neo4j vector
  index query consumer), todo-068 Graphiti audit, todo-069 MemPalace
  audit, todo-070 Paper/DocSection/ProjectMemory ingestion prereq.
- Shipped B3 in `scripts/kg/auto_refresh.sh`: when
  `KG_NEO4J_AUTO_UP=1`, the Stop-hook refresh warm-starts `make kg-up`
  if Neo4j isn't already running. Default off; opt-in only.
- Shipped B4 in `scripts/kg/compact_search.jq`: filter now accepts both
  the current bare-array kg-search output and a future
  `{results, search_mode, mode_reason, query_fixups}` wrapper. Today
  it prints `# search_mode: lexical_only  (reason: kg-search has not
  yet adopted hybrid output (todo-067 pending))` as a footer so agents
  always know which retrieval path produced the hits.

## Verification
- `python3 scripts/agents_db.py validate` -> passed with todo-066..070.
- `make kg-search KG_QUERY='hierachical viewpoint' KG_LIMIT=3` -> still
  works; new mode footer surfaces.
- `make kg-search KG_QUERY='RRI' KG_LIMIT=3` -> unchanged ordering;
  footer prints correctly.
- `KG_NEO4J_AUTO_UP=1 bash scripts/kg/auto_refresh.sh` -> exit 0;
  no-op when Neo4j is already up; would dispatch `make kg-up` if down
  (verified branch logic; not exercised because Neo4j is currently up).
- Raw Cypher smoke (not committed): vector query with ollama-computed
  embedding returns 5 nearest neighbors with cosine ~0.74. Confirms
  Phase B logic is sound; only thing missing is the Rust glue
  (todo-067) and the paper-ingest gap (todo-070).

## Canonical State Impact
None. All changes are agents-db backlog records, one helper script
edit (auto_refresh.sh), one jq filter edit (compact_search.jq), and a
debrief. `DECISIONS.md`, `PROJECT_STATE.md`, `OPEN_QUESTIONS.md`, and
`GOTCHAS.md` are unchanged.

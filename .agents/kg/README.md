# ARIA-NBV KG Profile

`.configs/litkg.toml` is the operator entrypoint for ARIA-NBV knowledge-graph ingestion.

Default representation:

- Graphify-style Markdown/JSON under `.agents/kg/generated/literature` is the primary durable output.
- Neo4j export bundles under `.agents/kg/generated/neo4j-export` are the default graph-traversal representation.
- CodeGraphContext indexes `aria_nbv/aria_nbv` into the local Neo4j runtime for symbol-level Python graph queries.
- Graphiti is optional for temporal doc/memory ingestion and is disabled by default in the TOML profile.
- MemPalace remains separate through `make memory-mine`; use it for repo memory retrieval, not as the structural literature/code KG backend.

Useful commands:

```bash
make kg-sync
make kg-materialize
make kg-semantic-enrich
make kg-export-neo4j
make kg-index-code
make kg-ingest-docs
make kg-ingest-papers
```

Generated database-like artifacts under `.agents/kg/` are covered by Git LFS patterns in `.gitattributes`.

---
id: gotchas
updated: 2026-05-15
scope: repo
owner: jan
status: active
tags: [workflow, training, cache, frames]
---

# Gotchas

## Environment and Tooling
- Prefer `uv run --project aria_nbv pytest` or `aria_nbv/.venv/bin/python -m pytest`; the system interpreter may miss dependencies such as `power_spherical`.
- Assume the environment is working unless the user indicates otherwise, but verify the exact interpreter before concluding a dependency problem.
- `make context` refreshes the lightweight routing artifacts only; use targeted search on `source_index.md`, `literature_index.md`, and `data_contracts.md` instead of loading broad dumps.
- `make context-heavy` and the `context-uml`, `context-docstrings`, or `context-tree` targets are explicit fallback tools for architecture or refactor tasks.
- Local CUDA visibility does not imply PyTorch3D CUDA support. On 2026-05-13 the workstation saw an RTX 3080 Ti, but PyTorch3D rasterization and point-mesh distance raised `RuntimeError: Not compiled with GPU support`; on 2026-05-15 PyTorch3D CUDA rasterization was restored by rebuilding from the activated `aria-nbv` mamba toolchain. Keep the mamba toolchain env active, `CUDA_HOME=$CONDA_PREFIX`, `FORCE_CUDA=1`, and `TORCH_CUDA_ARCH_LIST=8.6` when reinstalling CUDA extensions.

## Training and Validation
- Validation is disabled by default unless `trainer_config.enable_validation=true`; otherwise Lightning forces `limit_val_batches=0` and `check_val_every_n_epoch=0`.
- Treat explicit user termination criteria as binding. If they imply stronger verification, expand the test plan rather than stopping at a smoke test.
- Prefer real-data or integration-style verification when feasible for package changes; do not rely only on mocks for end-to-end behavior claims.

## VIN Offline Stores and Splits
- The canonical offline training path is `VinOfflineDataset`, configured through `VinOfflineSourceConfig` with `kind = "offline"`.
- Offline store splits are file-backed NumPy arrays under `splits/`; rebuild stale stores with `VinOfflineWriter` rather than legacy cache-index repair helpers.
- `sample_index.jsonl` is the source of scene/snippet coverage and shard rows.
- `VinOracleBatch.collate` expects model-ready `VinSnippetView` instances rather than raw `EfmSnippetView` samples.

## Frames and Geometry
- Pose-frame consistency and CW90 corrections are easy to misuse across rendering and VIN inputs.
- Use `PoseTW` and `CameraTW` instead of raw matrices in normal package code.
- Document tensor shapes and coordinate frames when a contract is not obvious from the type alone.

## EVL / OBB
- EVL OBB outputs are not batch-collatable yet; entity-aware runs may need `batch_size=None` or OBB outputs disabled.
- Candidate validity heuristics and semidense visibility proxies are conservative; do not assume they are equivalent to training masks unless the training loop explicitly applies them.
- Full-scene RRI during target-rollout generation is much more expensive than target-cropped RRI. Keep scene RRI as an explicit audit option even when PyTorch3D CUDA is available; target RRI remains the thesis-core label.

## Config and Pydantic
- `Field(default=<callable>)` stores the callable itself; use `Field(default_factory=...)` for computed defaults.
- Prefer config-local `field_validator` and `model_validator` hooks for cross-field validation and coercion.

## litkg / Knowledge Graph
- The Neo4j runtime now contains embedded content and code rows, including `Paper`, `PaperSection`, `DocSection`, `ProjectMemory`, generated context, backlog, and CodeGraphContext symbols. If semantic search looks code-only, recheck `make kg-status`, Neo4j, and the embedding index before assuming the old `todo-070` paper-coverage blocker has returned.
- Vector index is named `kg_embedding_index_2560` over `KGEmbeddingNode.kg_embedding` (dim 2560, cosine). Created idempotently by `enrich_embeddings.py:502`. The index name is dimension-suffixed; do not hardcode `node_embedding` or similar.
- Neo4j default credentials are `neo4j:litkglocal` per `.agents/external/litkg-rs/.env.example`. HTTP at `http://127.0.0.1:7474`, Bolt at `bolt://127.0.0.1:7687`.
- Ollama tunnel expected at `http://127.0.0.1:11434` (Mac runs `ollama serve`; reverse-tunnel via `ssh -N -R 11434:127.0.0.1:11434 ubuntu`). All embedding/chat-backed kg operations gracefully skip when the tunnel is down — they never fail loudly.
- Exported `Paper` nodes carry source provenance, and both lexical-only AND hybrid Neo4j vector hits inherit the configured `[authority_tiers]` since 2026-05-13 (todo-072 resolved). Hybrid hits with `repo_path` set on the vector node get a real `authority` label (`canonical`/`active`/etc.); vector hits without `repo_path` fall back to the synthetic `semantic` placeholder.
- Curated literature now reaches `kg-claim-check`'s top_sources via three coordinated changes (todo-073 resolved 2026-05-13): canonical-tier paths survive the `configured_source_paths` 512-cap first; `truncate_spans_to_budget` reserves ~2400 tokens for canonical spans; `top_sources` does a two-pass admission that rescues up to 4 canonical-tier spans beyond the standard cap of 8. `apply_confidence_floor` exempts canonical entries.
- `kg-claim-check`'s verdict heuristic (`classify_claim_span`) is narrow: it only flips to `Supports` for three hand-coded shapes (V0-GT-OBB sanity, V1-actor-visible contradiction, Habitat-as-main-simulator). Diverse claims (e.g. "VIN-NBV introduces RRI") fall through to a strict `overlap>=0.45` threshold that's unreachable in a 3-line span. Tracked as `todo-074`. Until it lands, treat unverifiable verdicts on novel claim shapes as a heuristic gap, not a data gap.
- `neo4j-cypher` MCP profile (`make kg-mcp-install`) gives agents a typed-query escape hatch when the wrappers don't cover the question (multi-hop joins, vector index with filters). Defaults to read-only; APOC is required for `get_neo4j_schema` and is loaded in the repo's Neo4j docker. See `litkg_quick_reference.md` for activation + example queries.
- `kg-search`, `kg-route`, `kg-claim-check` produce compact output by default since `c861d46`. Use `KG_VERBOSE=1` for full text or `KG_FORMAT=json` for raw JSON. Don't read the verbose form by default — it's >1000 lines.
- `kg find` now supports BM25/fuzzy lexical ranking plus optional hybrid Neo4j vector search. Use `--lexical-only` when debugging deterministic authority-tier behavior; default hybrid search can surface semantically close but stale/archive hits until authority is threaded into vector hits.
- `make kg-claim-check` excludes code and active-backlog spans from verdict evidence, then trusts authority labels `canonical | active` or source types `canonical_memory | docs`. The hybrid label `semantic` is not trusted for verdict support.

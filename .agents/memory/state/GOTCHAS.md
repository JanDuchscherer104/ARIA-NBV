---
id: gotchas
updated: 2026-05-13
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
- Local CUDA visibility does not imply PyTorch3D CUDA support. On 2026-05-13 the workstation saw an RTX 3080 Ti, but PyTorch3D rasterization and point-mesh distance raised `RuntimeError: Not compiled with GPU support`; rollout microset configs therefore used CPU until a CUDA-enabled PyTorch3D build/container is verified.

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
- Full-scene RRI during target-rollout generation is much more expensive than target-cropped RRI on the current CPU-only PyTorch3D path. Keep scene RRI as an explicit audit option until a downsampled or GPU-backed path is verified; target RRI remains the thesis-core label.

## Config and Pydantic
- `Field(default=<callable>)` stores the callable itself; use `Field(default_factory=...)` for computed defaults.
- Prefer config-local `field_validator` and `model_validator` hooks for cross-field validation and coercion.

## litkg / Knowledge Graph
- The Neo4j runtime today contains **only CodeGraphContext code symbols** (1936 KGEmbeddingNodes; 1234 Function, 352 Module, 209 Class, 141 File). Paper / PaperSection / DocSection / ProjectMemory nodes exist in the JSONL export but have never been loaded into the live DB. Vector queries return only code hits until `todo-070` lands.
- Vector index is named `kg_embedding_index_2560` over `KGEmbeddingNode.kg_embedding` (dim 2560, cosine). Created idempotently by `enrich_embeddings.py:502`. The index name is dimension-suffixed; do not hardcode `node_embedding` or similar.
- Neo4j default credentials are `neo4j:litkglocal` per `.agents/external/litkg-rs/.env.example`. HTTP at `http://127.0.0.1:7474`, Bolt at `bolt://127.0.0.1:7687`.
- Ollama tunnel expected at `http://127.0.0.1:11434` (Mac runs `ollama serve`; reverse-tunnel via `ssh -N -R 11434:127.0.0.1:11434 ubuntu`). All embedding/chat-backed kg operations gracefully skip when the tunnel is down — they never fail loudly.
- `paper:*` synthetic nodes lose `ParsedPaper.provenance` during ingest. Their `source_path` is null, blocking `[authority_tiers]` glob-based promotion. Workaround: lexical-search the citation key (`make kg-search KG_QUERY='hestia-lu2026'`) or the on-disk QMD path (`docs/contents/literature/*.qmd`).
- `kg-search`, `kg-route`, `kg-claim-check` produce compact output by default since `c861d46`. Use `KG_VERBOSE=1` for full text or `KG_FORMAT=json` for raw JSON. Don't read the verbose form by default — it's >1000 lines.
- `kg find` (the underlying CLI verb) does **pure lexical token matching** — no stemming, no synonyms, no embedding similarity. Typos collapse to single-token searches and lose. Filed as `todo-066`/`todo-067`.
- `make kg-claim-check` accepts only authority labels `canonical | active` for verdict counting (ranking.rs:46-54, context_pack.rs:1727). Labels below 1.2 multiplier are skipped even on exact lexical match.

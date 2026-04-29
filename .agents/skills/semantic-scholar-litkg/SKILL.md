---
name: semantic-scholar-litkg
description: Use when designing, implementing, reviewing, or operating Semantic Scholar, external-doc, local-source, or agent-memory ingestion for ARIA-NBV's litkg-rs knowledge graph, including TOML corpus configs, graphify/Graphiti/Neo4j/code-index/mempalace backend choices, MarkItDown URL ingestion, Context7 library-doc links, Python AST symbol extraction, and LFS handling for persisted KG/database artifacts.
---

# Semantic Scholar litkg

## Grounding

Start in ARIA-NBV, then hand implementation to the litkg-rs submodule.

1. Read `docs/typst/paper/main.typ`, this repo's `AGENTS.md`, and `.agents/memory/state/PROJECT_STATE.md`.
2. Read `.agents/external/litkg-rs/AGENTS.md`, `README.md`, `docs/architecture.md`, and `docs/kg-stack.md`.
3. For toolkit edits inside litkg-rs, use `.agents/external/litkg-rs/.agents/skills/litkg-rs/SKILL.md`.
4. Read `references/integration-spec.md` when the task touches source coverage, TOML shape, backend selection, or Semantic Scholar behavior.

## Defaults

- Keep implementation in `.agents/external/litkg-rs`; keep ARIA-NBV-specific assumptions in TOML configs, docs, or this skill.
- Prefer existing adapters and libraries before adding new code: litkg-rs pipeline primitives, CodeGraphContext/code-index, Graphiti, graphify, Neo4j export bundles, Context7, MarkItDown, and `mempalace-rs`.
- Treat `graphify` file output as the default durable sink, Neo4j export as optional query/runtime output, Graphiti/code-index as local enrichment, and mempalace as optional temporal memory/KG integration when its API fits cleanly.
- Drive inclusion/exclusion, parser options, external URL ingestion, and backend toggles from TOML.
- Track generated KG database files with Git LFS before committing them; do not commit local runtime cache directories.

## Workflow

1. Localize the requested graph capability to one of: literature metadata, local code symbols, repo docs, agent memory, external library docs, optional HTTPS pages, or backend export.
2. Inspect litkg-rs for an existing module or adapter boundary before adding a new crate or schema field.
3. Update litkg-rs first for repo-independent behavior; add ARIA-NBV config/docs only after the toolkit surface exists.
4. For Semantic Scholar, verify the current official API docs before changing request fields, pagination, headers, or rate-limit behavior.
5. For Context7, resolve the library id first, then store only durable doc links or provenance in litkg outputs unless the user asks to snapshot docs.
6. For MarkItDown, keep HTTPS ingestion optional and isolated behind a config flag because remote page conversion is slower and less reproducible than local file parsing.
7. Update `AGENTS.md` or `.agents/AGENTS_INTERNAL_DB.md` when a new durable ingestion surface becomes mandatory agent behavior.

## Validation

- In ARIA-NBV after skill or guidance edits: `python3 /home/jd/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/semantic-scholar-litkg` and `make check-agent-memory`.
- In litkg-rs after toolkit edits: `cargo fmt --all`, `cargo test`, and `make agents-db`.
- For KG runtime changes: run the narrow litkg-rs smoke target or check mode before starting long indexing jobs.

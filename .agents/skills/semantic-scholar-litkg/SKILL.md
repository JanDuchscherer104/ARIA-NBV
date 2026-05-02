---
name: semantic-scholar-litkg
description: Use when designing, implementing, reviewing, or operating Semantic Scholar, external-doc, local-source, or agent-memory ingestion for ARIA-NBV's litkg-rs knowledge graph, including TOML corpus configs, graphify/Graphiti/Neo4j/code-index/mempalace backend choices, MarkItDown URL ingestion, Context7 library-doc links, Python AST symbol extraction, and LFS handling for persisted KG/database artifacts.
---

# Semantic Scholar litkg

## Grounding

Start in ARIA-NBV, then hand implementation to the litkg-rs submodule.

1. Read `docs/typst/seminar_paper/main.typ`, this repo's `AGENTS.md`, and `.agents/memory/state/PROJECT_STATE.md`.
2. Read `.agents/external/litkg-rs/AGENTS.md`, `README.md`, `docs/architecture.md`, and `docs/kg-stack.md`.
3. For toolkit edits inside litkg-rs, use `.agents/external/litkg-rs/.agents/skills/litkg-rs/SKILL.md`.
4. Read `references/integration-spec.md` when the task touches source coverage, TOML shape, backend selection, or Semantic Scholar behavior.

## Defaults

- Keep implementation in `.agents/external/litkg-rs`; keep ARIA-NBV-specific assumptions in TOML configs, docs, or this skill.
- Prefer existing adapters and libraries before adding new code: litkg-rs pipeline primitives, the native litkg-rs Semantic Scholar REST client, CodeGraphContext/code-index, Graphiti, graphify, Neo4j export bundles, Context7, MarkItDown, and `mempalace-rs`.
- Treat litkg as ARIA-NBV's structural project KG and retrieval layer:
  `graphify`/JSONL are the durable default artifacts, Neo4j export is the
  optional traversal/runtime database, Graphiti is optional temporal ingestion,
  CodeGraphContext/code-index is code-symbol enrichment, and MemPalace remains a
  separate episodic memory-mining path through `make memory-mine`.
- Use `aria-litkg-memory` for ordinary broad task retrieval, claim checks, and
  consolidation routing. Use this skill when modifying the KG implementation,
  source coverage, backend contracts, Semantic Scholar behavior, or litkg-rs.
- Drive inclusion/exclusion, parser options, external URL ingestion, and backend toggles from TOML.
- Track generated KG database files with Git LFS before committing them; do not commit local runtime cache directories.
- Use one shared Codex/Gemini contract: `litkg capabilities --format json` for
  backend readiness and `litkg context-pack --format json|text` for
  task-specific action packs. Do not create separate agent workflows unless the
  CLI contract cannot express the need.

## Agent Contract

For broad cross-surface KG, docs, code, or thesis tasks, start with:

```bash
cargo run --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-cli -- context-pack \
  --config .configs/litkg.toml \
  --repo-root . \
  --task "<task>" \
  --profile thesis-coding \
  --format text
```

Use JSON when another agent or script will consume the result:

```bash
cargo run --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-cli -- context-pack \
  --config .configs/litkg.toml \
  --repo-root . \
  --task "<task>" \
  --profile docs-paper-sync \
  --format json
```

Check backend readiness without mutating tracked files:

```bash
cargo run --manifest-path .agents/external/litkg-rs/Cargo.toml -p litkg-cli -- capabilities \
  --config .configs/litkg.toml \
  --repo-root . \
  --format json
```

Prefer the focused profiles when the task is narrower than general thesis
coding: `docs-paper-sync`, `rri-oracle`, `vin-baseline`, or
`rollout-planning`. The context pack should give the smallest useful next step,
evidence spans, active backlog, missing leaves with repair commands, risk flags,
and verification commands.

## Workflow

1. Localize the requested graph capability to one of: literature metadata, local code symbols, repo docs, agent memory, external library docs, optional HTTPS pages, or backend export.
2. Inspect litkg-rs for an existing module or adapter boundary before adding a new crate or schema field.
3. Update litkg-rs first for repo-independent behavior; add ARIA-NBV config/docs only after the toolkit surface exists.
4. For Semantic Scholar, use the litkg-rs REST-native commands first: `s2 enrich`, `s2 search`, `s2 paper`, and `s2 recommend`. Verify the current official API docs before changing request fields, pagination, headers, or rate-limit behavior.
5. For Context7, resolve the library id first, then store only durable doc links or provenance in litkg outputs unless the user asks to snapshot docs.
6. For MarkItDown, keep HTTPS ingestion optional and isolated behind a config flag because remote page conversion is slower and less reproducible than local file parsing.
7. Update `AGENTS.md` or `.agents/AGENTS_INTERNAL_DB.md` when a new durable ingestion surface becomes mandatory agent behavior.

## Agents DB Integration

ARIA-NBV treats `.agents/issues.toml` and `.agents/todos.toml` as KG-ingested
agent-memory sources. When litkg-rs context packs touch backlog work, they
should preserve the DB's `context` and `references` fields instead of reducing
records to titles only.

Use this integration pattern:

- DB records cite internal sources with `repo:...`, papers with `bib:...`,
  durable paper identifiers with `arxiv:`, `doi:`, or `s2:`, external docs with
  `url:...`, Context7 docs with `context7:...`, and litkg evidence with
  `litkg:...`.
- `context-pack --format json` is the machine contract for carrying active
  backlog, evidence spans, relevant symbols, relevant papers, missing leaves,
  and verification commands into another agent or script.
- If the context pack omits active ARIA-NBV backlog rows, inspect the
  litkg-rs backlog adapter before widening ARIA-NBV-specific config. The
  adapter should accept both singular client tables (`[[issue]]`, `[[todo]]`)
  and legacy plural tables (`[[issues]]`, `[[todos]]`), plus `description` or
  `summary` text.
- Semantic Scholar enrichment should add identifiers and metadata to the KG
  registry; DB records should still cite the local `bib:` key first when the
  paper is already in `docs/references.bib`.

## Validation

- In ARIA-NBV after skill or guidance edits: `python3 /home/jd/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/semantic-scholar-litkg` and `make check-agent-memory`.
- In litkg-rs after toolkit edits: `cargo fmt --all`, `cargo test`, and `make agents-db`.
- For KG runtime changes: run the narrow litkg-rs smoke target or check mode before starting long indexing jobs.

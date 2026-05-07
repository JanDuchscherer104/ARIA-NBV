# litkg Agent Quick Reference

Use this reference for ARIA-NBV KG-backed retrieval and claim checks. LitKG is
a probationary source-backed router, claim checker, and research-memory layer,
not the default brain for every agent task. Use `semantic-scholar-litkg` only
when changing litkg-rs, KG source coverage, or KG operation itself.

## Probation Lane

Use LitKG when work crosses source families or needs source-backed arbitration:

- thesis writing and advisor-facing claims;
- literature synthesis and research framing;
- active backlog lookup and consolidation proposals;
- scaffold cleanup that needs stale-reference or duplicate-owner discovery;
- Rerun, plotting, or coding tasks that need current visual/data contracts
  across skills, docs, code, and memory.

Do not require LitKG for localized one-file coding edits, narrow test fixes, or
obvious file discovery. Start those with the nearest `AGENTS.md`, owning skill,
`aria-nbv-context`, and targeted `rg`/file reads. If LitKG returns broad or
generic context, treat it as advisory and inspect the concrete owner directly.

## Default Commands

- Route a broad task:
  `make kg-route KG_TASK="<task>"`
- Search indexed context quickly:
  `make kg-search KG_QUERY="<terms>"`
- Claim-check advisor-facing or thesis claims:
  `make kg-claim-check KG_CLAIM="<claim>"`
- Propose memory/backlog consolidation:
  `make kg-consolidate`
- Inspect source/backend readiness:
  `make kg-capabilities KG_FORMAT=json`

Use `KG_FORMAT=json` when another tool, script, or agent consumes the output.

## Expected Context-Pack Fields

Agent-facing context packs expose both the legacy fields and the newer compact
routing fields. Treat the compact fields as the first read, then open cited
spans before changing code or docs.

- `verb`
- `task_summary`
- `assumptions`
- `top_sources` with path, title, role, authority, freshness, source span,
  source type, scores, and why relevant
- `required_reads` derived from the highest-ranked deduped source paths
- `active_backlog`, `active_issues`, and `active_todos` with issue/todo id,
  priority, context, references, acceptance, and verification when present
- `risk_flags`
- `suggested_next_action` as an object with `summary`, optional `skill`,
  optional `command`, and `why`
- `verification_commands`
- `missing_context`, plus legacy aliases `missing_leaves` and
  `missing_context_leaves`

The following legacy fields remain for compatibility: `action_plan`,
`evidence_spans`, `missing_leaves`, `missing_context_leaves`, `active_issues`,
and `active_todos`.

Authority is configured in `.configs/litkg.toml`. Current canonical memory,
current thesis QMDs, thesis proposal Typst, and implementation code should rank
above historical seminar paper evidence and episodic history for comparable
matches.

Transcript-derived context follows a trust ladder: raw agent messages are
private low-trust search material, raw user messages are higher-signal but not
canonical, and distilled user intent plus later agent-grounded agreement is a
high-trust candidate memory. Only checked-in canonical memory, backlog, docs,
and code become current truth.

## Advanced Commands

- Emit a full JSON context pack:
  `make kg-route KG_TASK="<task>" KG_FORMAT=json`
- Use a specific context-pack profile directly:
  `.agents/external/litkg-rs/target/debug/litkg-cli context-pack --config .configs/litkg.toml --repo-root . --task "<task>" --profile thesis-coding --format json`
- Inspect a broad source/backlog query:
  `make kg-search KG_QUERY="<terms>" KG_FORMAT=json KG_LIMIT=10`

Question-answer synthesis is deferred until there is a real synthesis layer.
Use `kg-search` for fast retrieval and `kg-route` when an agent needs a context
pack with source ranking, backlog, risks, and verification.

## Fallback

If litkg is stale, unavailable, or returns broad/noisy context:

1. Use `aria-nbv-context` and targeted `rg`/file reads for local discovery.
2. Continue localized one-file or one-surface work when enough evidence exists.
3. Record or amend KG/backlog debt only when the litkg failure blocks the task or
   exposes durable scaffold drift.

Optional backend gaps such as Context7, Graphiti, Semantic Scholar enrichment,
OpenAI docs leaves, CodeGraphContext, or stale generated exports are warnings by
default. They become blockers only when the task explicitly requires that leaf
or no concrete local/canonical source can answer the request.

## Mandatory Claim Checks

Run `kg-claim-check` for advisor-facing proposal claims, thesis roadmap or
research-question claims, and literature-synthesis conclusions. Do not require
it for small internal wording, navigation, or localized implementation edits.

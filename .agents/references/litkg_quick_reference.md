# litkg Agent Quick Reference

Use this reference for ARIA-NBV KG-backed retrieval and claim checks. Use
`semantic-scholar-litkg` only when changing litkg-rs, KG source coverage, or KG
operation itself.

## Default Commands

- Route a broad task:
  `make kg-route KG_TASK="<task>"`
- Retrieve context for a question:
  `make kg-query KG_QUERY="<question>"`
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

Agent-facing context packs should expose, or be interpreted as if they expose:

- `task_summary`
- `assumptions`
- `top_sources` with path, title, authority, freshness, and why relevant
- `required_reads`
- `active_backlog` with issue/todo id, priority, acceptance, and verification
- `risk_flags`
- `suggested_next_action`
- `verification_commands`
- `missing_context`

Until litkg-rs exposes all fields directly, agents must inspect cited canonical
sources before treating retrieved statements as current truth.

## Fallback

If litkg is stale, unavailable, or returns broad/noisy context:

1. Use `aria-nbv-context` and targeted `rg`/file reads for local discovery.
2. Continue localized one-file or one-surface work when enough evidence exists.
3. Record or amend KG/backlog debt only when the litkg failure blocks the task or
   exposes durable scaffold drift.

## Mandatory Claim Checks

Run `kg-claim-check` for advisor-facing proposal claims, thesis roadmap or
research-question claims, and literature-synthesis conclusions. Do not require
it for small internal wording, navigation, or localized implementation edits.

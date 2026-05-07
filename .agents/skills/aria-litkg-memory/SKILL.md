---
name: aria-litkg-memory
description: Use for KG-backed ARIA-NBV retrieval, task routing, claim checks, current-truth checks, backlog lookup, and consolidation proposals.
metadata:
  mode: router
  not_when:
    - "local file discovery alone is enough"
    - "litkg-rs implementation, KG config, or backend contracts are changing"
    - "a concrete failure loop owns the task"
  handoff_to:
    - "aria-nbv-context for local-only discovery"
    - "semantic-scholar-litkg for litkg-rs, KG config, or backend edits"
    - "diagnose-aria for KG ingestion failures"
  evidence_required:
    - "litkg command output or cited KG result"
    - "canonical source inspection before promoting retrieved truth"
    - "claim-check command for advisor-facing claims"
  applies_to:
    - "**"
  triggers:
    - "kg-route"
    - "claim check"
    - "source-backed"
    - "consolidate memory"
  must_read:
    - "AGENTS.md"
    - ".agents/references/source_order.md"
    - ".agents/references/litkg_quick_reference.md"
  verification:
    - "make kg-capabilities KG_FORMAT=json"
    - "make kg-route KG_TASK=\"<task>\" KG_FORMAT=json"
    - "make kg-claim-check KG_CLAIM=\"<claim>\""
---

# ARIA litkg Memory

Use this skill when litkg should act as a probationary source-backed router,
claim-check layer, or research-memory retrieval surface for work that crosses
source families.

## Protocol

1. Read `AGENTS.md`, `.agents/references/source_order.md`, and
   `.agents/references/litkg_quick_reference.md`.
2. Check backend/source readiness with `make kg-capabilities KG_FORMAT=json`
   when freshness matters.
3. Use `make kg-route KG_TASK="<task>"` for broad task routing.
4. Use `make kg-search KG_QUERY="<terms>"` for fast retrieval, or `kg-route`
   when the agent needs a full context pack.
5. Use `make kg-claim-check KG_CLAIM="<claim>"` for advisor-facing proposal,
   roadmap, research-question, or literature-synthesis claims.
6. Inspect cited canonical sources before treating retrieved statements as
   current truth.
7. Use `make kg-consolidate` for proposal-style memory/backlog updates; do not
   silently promote episodic notes.

## Source Authority

Until litkg retrieval exposes explicit authority/freshness metadata everywhere,
rank sources with `.agents/references/source_order.md` and inspect cited
canonical sources before treating retrieved statements as current truth.

## Fallback

If litkg is stale, unavailable, or too noisy for a localized task, fall back to
`aria-nbv-context` plus targeted file reads. Record KG debt only when the failure
blocks the task or exposes durable scaffold drift.

## Verification

- `make kg-capabilities KG_FORMAT=json`
- `make kg-route KG_TASK="<task>" KG_FORMAT=json`
- `make kg-claim-check KG_CLAIM="<claim>"` for advisor-facing claims
- `make check-agent-memory` after non-trivial memory or guidance changes

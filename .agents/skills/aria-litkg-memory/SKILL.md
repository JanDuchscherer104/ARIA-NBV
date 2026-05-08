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
2. Default first read: `make kg-search KG_QUERY="<terms>"` for fast,
   score-ranked retrieval over code/docs/memory/backlog/literature. Empirical
   verdict: this is the killer verb for localized lookups.
3. Escalate to `make kg-route KG_TASK="<task>"` only when the agent needs a
   full context pack (top_sources + required_reads + active_backlog +
   verification_commands + missing_context).
4. Use `make kg-claim-check KG_CLAIM="<claim>"` for advisor-facing proposal,
   roadmap, research-question, or literature-synthesis claims; expect
   `verdict, confidence, supporting_evidence, contradicting_evidence`.
5. When KG output looks degraded or empty, run `make kg-status` first as a
   fast 0/1 health probe. If non-zero, fall back to `aria-nbv-context` plus
   targeted reads and record the outage in the debrief.
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

- `make kg-status` first as a fast 0/1 probe; if non-zero, fall back to
  `aria-nbv-context` plus targeted reads and record the KG outage in the
  debrief instead of waiting for the heavier commands below.
- `make kg-capabilities KG_FORMAT=json`
- `make kg-search KG_QUERY="<terms>" KG_FORMAT=json` for retrieval verification.
- `make kg-route KG_TASK="<task>" KG_FORMAT=json` for context-pack verification.
- `make kg-claim-check KG_CLAIM="<claim>" KG_FORMAT=json` for advisor-facing
  claims; expect `verdict` and `confidence` populated.
- `make check-agent-memory` after non-trivial memory or guidance changes.

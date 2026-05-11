---
id: 2026-05-11_prune_vestigial_kg_verbs_and_file_lean_payload_todo
date: 2026-05-11
title: "prune vestigial kg verbs and file lean payload todo"
status: done
topics: [litkg, makefile, signal-noise, audit]
confidence: high
canonical_updates_needed: []
---

## Task
After P1+P2 shipped, audit the rest of litkg's surface for sloppy or
unpragmatic noise. Prune what is safe to delete now and file the larger
clean-ups (Rust changes, stale-file audit) as backlog.

## Method
1. Inspected the raw `kg-route KG_TASK='smoke' KG_FORMAT=json` payload to
   measure field-level signal vs noise. Full payload is 242 KB; trimming
   six low-value fields drops it to 39 KB (6x).
2. Inventoried all 24 `kg-*` Makefile targets and classified into:
   agent-facing (5), operator-day-1 (8), ingestion plumbing (9), and
   vestigial/duplicate (4).
3. Probed each of the four less-used verbs (`kg-brief`, `kg-related`,
   `kg-consolidate`, `kg-show-paper`) to compare schemas:
   - `kg-brief` returns an identical ContextPack keys list to `kg-route`
     (same context-pack endpoint with a "brief: " task-prefix). Same
     duplicate-alias anti-pattern as the already-deleted `kg-query`.
   - `kg-related` returns an array of search-result objects, not a
     ContextPack. Functionally a subset of `kg-search` with a path query.
   - `kg-consolidate` and `kg-show-paper` return distinct, useful shapes;
     kept.
4. Wrote `Plan - Audit + prune remaining litkg noise` in the plan file
   (see plan log). User approved.
5. Shipped F1: deleted vestigial verbs, fixed stale comment, collapsed
   smoke target into a flag. Filed F2/F3 as backlog.

## Findings
- Deleted from `Makefile`: `kg-brief`, `kg-related`, `kg-ingest-docs-smoke`
  targets and their `.PHONY` entries.
- `kg-ingest-docs` now accepts `KG_SMOKE=1` for the single-doc smoke pass
  (replaces the removed dedicated target).
- Fixed stale `kg-query` mention in `.configs/litkg.toml:313` to read
  `kg-search/kg-route/kg-claim-check`.
- Updated `.agents/references/operator_quick_reference.md:71` to use the
  new `make kg-ingest-docs KG_SMOKE=1` form.
- Documented an invariant in `scripts/kg/compact_route.jq`: the filter
  MUST NOT surface `evidence_spans`, `backend_status`, `action_plan`,
  `assumptions`, `missing_leaves`, `missing_context_leaves`, `profile`,
  `budget_tokens`, or `truncated`. This locks in the signal/noise contract
  established by P1.
- Filed three new agents-DB todos:
  - `todo-063`: Lean ContextPack default (Rust change). Acceptance:
    trivial smoke payload drops 242 KB to <=50 KB. Issue link: `issue-025`.
  - `todo-064`: Audit and prune stale `.agents/kg/generated/literature/`
    notes (21 MB total, many 2026-03-30 seminar-era).
  - `todo-065`: Confirm/remove `matched_field` and consider
    score-normalization for `kg-search`.
- Remaining vestigial-verb references in the repo are all archival:
  `.agents/work/` (gitignored review notes) and `.agents/resolved.toml`
  (history of the now-resolved todo that originally introduced the verbs).
  Left as-is.

## Verification
- `rg -n 'kg-brief|kg-related|kg-ingest-docs-smoke' Makefile` -> 0 hits.
- `rg -n 'kg-query' .configs/litkg.toml` -> 0 hits.
- `rg -n 'kg-brief|kg-related|kg-ingest-docs-smoke' .agents/skills/
  .agents/references/ CLAUDE.md AGENTS.md docs/AGENTS.md` -> 0 hits.
- `make kg-route KG_TASK='smoke' | wc -l` -> 14 (compact filter intact).
- `make kg-search KG_QUERY='RRI' KG_LIMIT=5 | wc -l` -> 13.
- `make kg-claim-check KG_CLAIM='...' | wc -l` -> 10.
- `python3 scripts/agents_db.py validate` -> passed (with new todo-063,
  todo-064, todo-065).
- `make check-agent-memory` -> passed.

## Canonical State Impact
None. All changes are Makefile recipes, config comment, helper-doc
updates, and backlog records. `DECISIONS.md`, `PROJECT_STATE.md`,
`OPEN_QUESTIONS.md`, and `GOTCHAS.md` are unchanged. Three new todos
(`todo-063`/`-064`/`-065`) capture deferred work against `issue-025`.

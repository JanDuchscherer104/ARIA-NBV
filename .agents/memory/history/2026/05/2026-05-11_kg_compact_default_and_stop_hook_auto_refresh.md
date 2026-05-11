---
id: 2026-05-11_kg_compact_default_and_stop_hook_auto_refresh
date: 2026-05-11
title: "kg compact default and stop hook auto refresh"
status: done
topics: [litkg, hooks, ollama, signal-noise]
confidence: high
canonical_updates_needed: []
---

## Task
Land P1 (compact-by-default kg-* output) and P2 (tunnel-aware Stop-hook auto
refresh) from the native-litkg-integration plan. P0 (paper-node provenance
fix) was pivoted to backlog after verification revealed it needs a real
litkg-rs Rust change, not a config edit.

## Method
1. V1/V2/V3 verification:
   - `ranking.rs:23-54` shows authority label assigned from numeric tier
     (>=1.5 canonical, >=1.2 active, <1.0 historical, else default).
   - `context_pack.rs:1727 is_canonical_or_active` accepts only
     `canonical|active` labels.
   - V2 (`grep paper:rri-theory nodes.jsonl | jq .properties`) confirmed
     all 84 `paper:*` nodes carry `source_path: null` and `repo_path: null`
     so no `[authority_tiers]` glob can promote them. P0 pivoted to backlog.
   - V3: kg-route JSON top-level keys captured; `top_sources[].scores`
     carries the canonical/active/default label as `.scores.authority`;
     `.required_reads[]` are objects `{path,title,reason}` not strings.
2. P1: wrote three jq filters under `scripts/kg/`:
   - `compact_route.jq` -> 14-line summary (task, verb, top 3 sources,
     active backlog ids, risk flags, next, read_first).
   - `compact_search.jq` -> 13-line summary (top 5 hits with path).
   - `compact_claim_check.jq` -> 10-line summary (claim, verdict +
     confidence, top 2 supporting/contradicting).
3. P1: modified `kg-search`, `kg-route`, `kg-claim-check` recipes in the
   Makefile to pipe JSON through the compact filter by default; opt out via
   `KG_VERBOSE=1` (full text) or `KG_FORMAT=json` (raw JSON).
4. P1: documented in `.agents/references/litkg_quick_reference.md` Output
   Modes section.
5. P2: wrote `scripts/kg/auto_refresh.sh` with four guards (ollama probe,
   staleness check, lock file, background spawn). Logs to
   `.agents/kg/.refresh.log`; honors `KG_FORCE=1`.
6. P2: added second Stop-hook command in `.claude/settings.json` (timeout
   2000ms) and `.codex/hooks.example.json` (parity).
7. P2: added `.refresh.log`, `.refresh.lock`, `.last-refresh` to
   `.gitignore`; documented in `.agents/references/operator_quick_reference.md`.
8. Filed `todo-062` against `issue-025` capturing the P0 deferral with
   acceptance/verification spelled out.
9. Updated `aria-litkg-memory/SKILL.md` step 4 with the known gap + manual
   `kg-search` fallback so agents stop trusting `unverifiable` verdicts on
   literature claims.

## Findings
- Compact output is 14 lines vs 1115 lines verbose for the same kg-route
  call - **80x reduction** in default signal/noise.
- `make kg-claim-check KG_CLAIM='...GT-OBB...V0 sanity/upper-bound'`
  already returns `supported` with confidence 1.0 against canonical memory;
  also the VIN-NBV/RRI claim now flips to `supported` via roadmap.qmd
  matches (the recent thesis-roadmap edits made it lexically discoverable).
- Edge case verified: `kg-route KG_TASK='zzzzz nonsense'` falls through to
  "no high-signal sources; consider aria-nbv-context" -> `aria-nbv-context`
  skill suggestion, 12 lines clean.
- `auto_refresh.sh` smoke-tested all four branches: tunnel-up real refresh
  (10s), no-changed-sources skip, force-bypass, lock collision (intercepted
  by staleness check first).
- `kg-refresh-light` completes in ~10s on this machine; well below the 2s
  Stop-hook timeout (because the script detaches before the work runs).

## Verification
- `python3 scripts/agents_db.py validate` - passed.
- `make check-agent-memory` - passed.
- `make kg-route KG_TASK='harden bounded oracle-RRI lookahead' | wc -l` -> 14.
- `make kg-route KG_TASK='...' KG_VERBOSE=1 | wc -l` -> 1115 (preserved).
- `make kg-route KG_TASK='...' KG_FORMAT=json | jq '.top_sources|length'`
  -> 8 (raw payload intact).
- `make kg-search KG_QUERY='VIN-NBV RRI' | wc -l` -> 13.
- `make kg-claim-check KG_CLAIM='...' | wc -l` -> 10.
- `bash scripts/kg/auto_refresh.sh` four branches all return exit 0 with
  the correct log line per branch.

## Canonical State Impact
None. All changes are Makefile recipes, helper scripts, skill docs, and
backlog records. `DECISIONS.md`, `PROJECT_STATE.md`, `OPEN_QUESTIONS.md`,
and `GOTCHAS.md` are unchanged. `todo-062` captures the deferred P0 work
against `issue-025`.

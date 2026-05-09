---
id: 2026-05-09_codex_transcript_decision_extraction
date: 2026-05-09
title: "Codex Transcript Decision Extraction"
status: done
topics: [codex, transcripts, memory, litkg]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
---

## Task
Implement repo-local extraction of ARIA-NBV Codex user messages and plan-mode
answers into KG-configured transcript memory artifacts.

## Method
Added `scripts/codex_transcript_extract.py` with dry-run/write modes, a
`codex-transcripts` make target, parser tests, and a dated transcript batch
under `.agents/memory/transcripts/`. The extractor now writes both heuristic
candidate distillates and conservative reviewed distillates so transcript
evidence is explicitly separated from canonical promotion.

## Findings
The default Codex store is `${CODEX_HOME:-$HOME/.codex}/sessions`, and this
machine also has a restored backup under
`/home/jd/Desktop/pre-essential-restore-20260425-234438/.codex/sessions`.
The extractor filters repo-cwd records, allows neutral-cwd marker fallback,
and rejects other checkout/worktree records so cross-repo scaffold sessions do
not pollute ARIA-NBV transcript memory.

## Verification
- `/home/jd/repos/ARIA-NBV/aria_nbv/.venv/bin/ruff check scripts/codex_transcript_extract.py aria_nbv/tests/agent_memory/test_codex_transcript_extract.py`
- `cd aria_nbv && uv run pytest tests/agent_memory/test_codex_transcript_extract.py`
- `aria_nbv/.venv/bin/python scripts/codex_transcript_extract.py`
- `make codex-transcripts CODEX_TRANSCRIPT_ARGS='--write --date 2026-05-09'`
- `make kg-status`
- `make kg-search KG_QUERY='transcript plan-mode answer raw_policy' KG_FORMAT=json KG_LIMIT=5`
- `make check-agent-memory`

## Canonical State Impact
Updated `.agents/memory/state/DECISIONS.md` with the Codex session location,
raw-transcript exclusion rule, and the definition of plan-mode transcript
answers. Also recorded that reviewed transcript status is routing metadata only
until accepted into canonical memory, backlog, docs, or code.

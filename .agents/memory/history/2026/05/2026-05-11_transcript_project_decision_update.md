---
id: 2026-05-11_transcript_project_decision_update
date: 2026-05-11
title: "Transcript Project Decision Update"
status: done
topics: [codex, transcripts, memory, decisions, litkg]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/transcripts/user/2026-05-11/user_messages.jsonl
  - .agents/memory/transcripts/user/2026-05-11/plan_mode_answers.jsonl
  - .agents/memory/transcripts/distilled/2026-05-11/candidate_decisions.jsonl
  - .agents/memory/transcripts/distilled/2026-05-11/reviewed_decisions.jsonl
  - .agents/memory/transcripts/distilled/2026-05-11/manifest.json
---

## Task

Refresh transcript-derived ARIA-NBV user-message and plan-answer artifacts for
2026-05-11, then promote only still-current durable project decisions into
canonical memory.

## Method

Ran `make codex-transcripts CODEX_TRANSCRIPT_ARGS='--write --date 2026-05-11'`
against the default Codex session store and restored backup store. Before
editing canonical decisions, checked current VIN/offline store code and
inspector docs/code for contradictions in padded tensor, candidate-mask, shard,
and native Rerun component wording.

## Findings

The 2026-05-11 transcript batch wrote sanitized user extracts, plan-mode answer
extracts, candidate distillates, reviewed distillates, and a manifest under
`.agents/memory/transcripts/`. The manifest counted 858 session files, 273
ARIA-NBV candidate sessions, 2,277 deduplicated user messages, 527
deduplicated plan-mode answers, and 2,418 candidate distillates. The
contradiction check found current support for `candidate_count`/length masking,
padded dense model-facing tensors, full candidate-width preservation,
manifest/sample-index/split/shard VIN offline stores, and native Rerun
camera/OBB logging. No SDV compatibility requirement was promoted; detailed
Rerun UI defaults stayed outside canonical decisions.

## Verification

Verification passed with `make check-agent-memory`,
`make agents-db AGENTS_ARGS='validate'`, and `make kg-status`. The required
transcript-focused `make kg-search KG_QUERY='rollout target crop VIN offline
Rerun native Boxes3D transcript decisions' KG_FORMAT=json KG_LIMIT=8` returned
canonical code and canonical memory ahead of transcript material. A narrower
rollout/store query returned `.agents/memory/state/DECISIONS.md` as the top two
hits, with docs/backlog evidence below canonical memory.

## Canonical State Impact

Updated `.agents/memory/state/DECISIONS.md` with compact durable project and
docs decisions mined from transcript evidence. Transcript artifacts remain
evidence until a candidate is accepted into canonical memory, docs, backlog, or
code.

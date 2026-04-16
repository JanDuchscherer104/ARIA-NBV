---
id: 2026-04-16_hook_first_context_guidance_cleanup
date: 2026-04-16
title: "Remove redundancy between hook output and hook-first startup guidance"
status: done
topics: [codex, hooks, context, scaffold]
confidence: high
canonical_updates_needed: []
files_touched:
  - path: AGENTS.md
    kind: guidance
  - path: .agents/skills/aria-nbv-context/SKILL.md
    kind: skill
  - path: .agents/skills/aria-nbv-context/scripts/nbv_context_index.sh
    kind: generator
  - path: .codex/hooks/run_make_context_clean.sh
    kind: hook-script
---

## Task

Clean up hook-first startup guidance so `AGENTS.md` owns the startup contract and
the hook message only describes the refreshed generated files.

## Method

Updated root guidance, context skill wording, and generated source-index
wording so `make context` is documented as manual refresh only. Simplified the
startup hook output to artifact descriptions and changed raw command logs to a
failure-only path.

## Verification

- `bash .codex/hooks/run_make_context_clean.sh`
- `make context`
- `make check-agent-scaffold`
- `make check-agent-memory`
- `uvx pre-commit run --all-files --show-diff-on-failure`

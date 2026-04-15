---
id: 2026-04-15_codex_startup_context_hook
date: 2026-04-15
title: "Enable Codex startup context refresh hook"
status: done
topics: [codex, hooks, context, scaffold]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - path: .codex/hooks.json
    kind: hook
  - path: .codex/hooks/run_make_context_clean.sh
    kind: hook-script
  - path: .agents/references/codex_hooks.md
    kind: reference
---

## Task

Enable a repo-local Codex startup hook that refreshes the lightweight generated
context bundle for new trusted Codex sessions.

## Method

Added `.codex/config.toml`, `.codex/hooks.json`, and a small shell wrapper that
runs `make context`, strips ANSI color codes, and prints a concise
agent-facing description of the generated context files.

## Verification

- `python3 -m json.tool .codex/hooks.json`
- `python3` TOML parse of `.codex/config.toml`
- `bash .codex/hooks/run_make_context_clean.sh`

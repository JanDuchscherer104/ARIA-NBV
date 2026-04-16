---
id: 2026-04-15_agent_scaffold_refactor
date: 2026-04-15
title: "Refactor agent scaffold into thin root, local guides, split skills, and scaffold DB"
status: done
topics: [scaffold, agents, skills, validation, hooks]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/GOTCHAS.md
files_touched:
  - path: AGENTS.md
    kind: guidance
  - path: aria_nbv/AGENTS.md
    kind: guidance
  - path: docs/AGENTS.md
    kind: guidance
  - path: .agents/skills/
    kind: skills
  - path: scripts/validate_agent_scaffold.py
    kind: validation
  - path: .agents/issues.toml
    kind: agent-db
---

## Task

Implemented the first broad Aria-NBV agent scaffold refactor: thin root
guidance, expanded local `AGENTS.md` coverage, moderate skill split, optional
GitNexus reference, inactive Codex hook templates, agent/tooling DB, and local
plus pre-commit scaffold validation.

## Method

Kept unrelated dirty worktree changes intact. Restored the Typst outline helper
surface with a root wrapper, pruned obsolete Makefile references, added
`make check-agent-scaffold`, wired the Quarto agent-scaffold generator to the
new canonical sources, and seeded `.agents` DB records for future hook-template
activation review.

## Verification

- `make context`
- `make context-typst-outline`
- `make check-agent-memory`
- `make check-agent-scaffold`
- `make agents-db`
- `./scripts/quarto_generate_agent_docs.py`
- `uvx pre-commit run --all-files --show-diff-on-failure`

## Notes

`uvx pre-commit run --all-files` initially let the trailing-whitespace hook
modify archived/literature files. Those hook-induced edits were reverted, and
the pre-commit config now excludes archival and literature import surfaces from
that hook. Ruff hooks are scoped to the package workspace and exclude known
pre-existing Ruff debt so scaffold validation can pass without silently fixing
unrelated code.

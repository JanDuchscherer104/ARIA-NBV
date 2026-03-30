---
id: 2026-03-29_repo_relative_paths_policy
date: 2026-03-29
title: "Adopt Repo-Relative Paths In Scaffold Guidance"
status: done
topics: [scaffold, codex, memory, path-conventions]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/OWNER_DIRECTIVES.md
files_touched:
  - AGENTS.md
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/OWNER_DIRECTIVES.md
---

# Task

Switched the active root scaffold guidance to repo-relative paths and recorded repo-relative paths as the preferred convention for scaffold guidance and canonical memory.

# Method

- Replaced absolute workspace path links in `AGENTS.md` with repo-relative path references.
- Added a root scaffold rule requiring repo-relative paths unless an external tool explicitly requires an absolute path.
- Recorded the same preference in canonical owner guidance and durable repo decisions.

# Verification

- `make check-agent-scaffold`
- `make check-agent-memory`

# Canonical State

- Updated `.agents/memory/state/OWNER_DIRECTIVES.md` and `.agents/memory/state/DECISIONS.md` to preserve the repo-relative path convention for future scaffold and memory edits.

## Prompt Follow-Through

- Captured the owner request to always use repo-relative paths and promoted it into canonical memory instead of leaving it only in chat.
- Applied the convention immediately to the active root `AGENTS.md`.

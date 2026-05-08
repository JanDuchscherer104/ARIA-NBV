---
description: Show ranked agents-db backlog and validate schema.
allowed-tools: Bash(make agents-db), Bash(make agents-db AGENTS_ARGS='validate')
argument-hint: "[validate]"
---

If $ARGUMENTS is "validate", run `make agents-db AGENTS_ARGS='validate'`.
Otherwise run `make agents-db` and surface only high-priority active records
plus any blocker. Reference IDs as `issue-NNN` / `todo-NNN` / `refactor-NNN`.

When recommending action on a record, first check whether a similar item is
already resolved in `.agents/resolved.toml` so we do not redo settled work.

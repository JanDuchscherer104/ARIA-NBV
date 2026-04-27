# Agent Memory Templates

Use these templates for native debriefs under `.agents/memory/history/YYYY/MM/`.

For non-trivial work, write a debrief record under `.agents/memory/history/YYYY/MM/`.

Existing records with `status: legacy-imported` are grandfathered archive evidence and do not need to be backfilled unless a task explicitly asks for it.

## Required Frontmatter
- `id`
- `date`
- `title`
- `status`
- `topics`
- `confidence`
- `canonical_updates_needed`

If no canonical state document changed, set `canonical_updates_needed: []`.

## Native Debrief With No Canonical Updates

```yaml
---
id: 2026-03-25_example_debrief
date: 2026-03-25
title: "Example Debrief"
status: done
topics: [scaffold, codex, memory]
confidence: high
canonical_updates_needed: []
---
```

## Native Debrief With Canonical Updates

```yaml
---
id: 2026-03-25_example_with_state_updates
date: 2026-03-25
title: "Example Debrief With State Updates"
status: done
topics: [scaffold, codex, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
---
```

## Optional Fields
- `files_touched`
- `source_legacy_path`
- `artifacts`
- `assumptions`

## Canonical State and Legacy Notes
- If the task changes current truth, update one or more files in `.agents/memory/state/` and list them in `canonical_updates_needed`.
- Otherwise keep `canonical_updates_needed: []`.
- Existing `status: legacy-imported` records are archive evidence; do not backfill them unless a task explicitly requires it.
- Legacy `.codex` notes were migrated into `.agents/memory/history/` and `archive/codex-legacy/`. Do not recreate `.codex` as a task-notes bucket.

Keep the body concise:
- task
- method or commands
- findings or outputs
- verification
- canonical state impact

Useful additions when they materially clarify the work:
- mention staged scope or commit scope when the worktree was dirty
- note whether compatibility was preserved deliberately or removed deliberately

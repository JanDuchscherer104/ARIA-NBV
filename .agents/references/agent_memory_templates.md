# Agent Memory Templates

Use these templates for native debriefs under `.agents/memory/history/YYYY/MM/`.

Existing records with `status: legacy-imported` are grandfathered archive evidence and do not need to be backfilled unless a task explicitly asks for it.

## Required Frontmatter
- `id`
- `date`
- `title`
- `status`
- `topics`
- `confidence`
- `canonical_updates_needed`

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

```md
## Prompt Follow-Through

No durable owner prompt items were present beyond the task-specific request.
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

```md
## Prompt Follow-Through

- Captured durable owner prompt guidance and promoted it into `.agents/memory/state/OWNER_DIRECTIVES.md`.
- Recorded any additional canonical state updates in `canonical_updates_needed`.
```

## Optional Fields
- `files_touched`
- `source_legacy_path`
- `artifacts`
- `assumptions`

Keep the body concise:
- task
- method or commands
- findings or outputs
- verification
- prompt follow-through
- canonical state impact

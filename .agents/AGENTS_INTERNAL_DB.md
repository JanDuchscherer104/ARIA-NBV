# ARIA-NBV Internal Agent DB

This file tracks the active engineering and research backlog for ARIA-NBV.

## Core Backlog

- **Issues**: [issues.toml](./issues.toml) - High-priority bugs and scaffold leaks.
- **TODOs**: [todos.toml](./todos.toml) - Immediate and research milestones.
- **Refactors**: [refactors.toml](./refactors.toml) - Structural and hygiene improvements.
- **Resolved**: [resolved.toml](./resolved.toml) - Archive of completed work.

## Relation to Memory

While `.agents/memory/state/` holds the **durable current truth**, this DB holds the **active maintenance debt**.

Active issues and todos must carry compact prose context plus structured
`references` pointers. Use `repo:` for internal files, `bib:` for papers in
`docs/references.bib`, durable identifiers such as `arxiv:`/`doi:`/`s2:`,
external docs as `url:`, Context7 library docs as `context7:`, and litkg-rs
evidence as `litkg:`. This keeps local backlog records auditable and
machine-usable by the litkg-rs context-pack/KG pipeline.

## Priority Pillars

1. **Scaffold Hygiene**: Fix stale skills and dense root guidance.
2. **Docs Discipline**: Triage `docs/` tree by role.
3. **Research Core**: Build entity-aware oracle RRI and the NBV Gym simulator.
4. **Agentic Lifecycle**: Port PR and issue lifecycle workflows.

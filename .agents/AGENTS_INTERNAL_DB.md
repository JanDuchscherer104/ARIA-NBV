# ARIA-NBV Internal Agent DB

This file tracks the active engineering and research backlog for ARIA-NBV.

## Core Backlog

- **Issues**: [issues.toml](./issues.toml) - High-priority bugs and scaffold leaks.
- **TODOs**: [todos.toml](./todos.toml) - Immediate and research milestones.
- **Refactors**: [refactors.toml](./refactors.toml) - Structural and hygiene improvements.
- **Resolved**: [resolved.toml](./resolved.toml) - Archive of completed work.

## Relation to Memory

While `.agents/memory/state/` holds the **durable current truth**, this DB holds the **active maintenance debt**.
Extracted proposal, transcript, or review requirements become agents-DB work
when they are actionable. They should become canonical state only when they
change current truth, and otherwise belong in dated memory debriefs.

Active issues and todos must carry compact prose context plus structured
`references` pointers. Use `repo:` for internal files, `bib:` for papers in
`docs/references.bib`, durable identifiers such as `arxiv:`/`doi:`/`s2:`,
external docs as `url:`, Context7 library docs as `context7:`, and litkg-rs
evidence as `litkg:`. This keeps local backlog records auditable and
machine-usable by the litkg-rs context-pack/KG pipeline.

## Priority Pillars

1. **Scaffold Hygiene**: Fix stale skills and dense root guidance.
2. **Docs Discipline**: Triage `docs/` tree by role.
3. **Research Core**: Freeze M1 contracts, build entity-aware target RRI,
   observed-only target selection, bounded oracle rollouts, and the
   finite-candidate candidate-query Transformer Q_H. Gym-style online
   simulators are stretch or bridge work after the ASE rollout/Q_H path is
   stable.
4. **Agentic Lifecycle**: Port PR and issue lifecycle workflows.

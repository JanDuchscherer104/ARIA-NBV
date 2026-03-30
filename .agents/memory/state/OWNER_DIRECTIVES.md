---
id: owner_directives
updated: 2026-03-29
scope: repo
owner: jan
status: active
tags: [codex, workflow, owner, directives]
---

# Owner Directives

Use this file for durable owner TODOs, issues, core requests, and workflow feedback that should shape future agent behavior across tasks. Do not use it as a task log or backlog dump.

## Memory Priority
- Treat explicit owner TODOs, issues, core requests, and feedback from prompts as first-class memory inputs.
- Promote durable owner guidance into canonical memory instead of leaving it only in chat or a dated debrief.
- Native debriefs must record prompt follow-through and name the canonical files that were updated.
- When the owner explicitly requests autonomous overnight iteration, keep chaining bounded simplify, verify, and report passes without asking for feedback until a concrete blocker requires a decision.

## Scaffold Preferences
- Keep the repo-root `AGENTS.md` compact, self-explanatory, and free of inline TODO/FIXME scaffolding debris.
- Prefer coherent, intuitive section titles and groupings in `AGENTS.md` files.
- Keep repo-wide routing and memory policy in the root scaffold and skills; keep implementation-specific rules in path-local `AGENTS.md` files.
- Use repo-relative paths in scaffold guidance, canonical memory, and debriefs unless a tool explicitly requires an absolute path.
- Prefer short trigger-and-effect descriptions for repo-specific commands over long instruction catalogs in the root scaffold.
- Make project-specific implementation patterns easy to discover in path-local guidance, especially the `BaseConfig` / `.setup_target()` factory pattern.

## Implementation Preferences
- Prefer simple, direct implementations that fit the current architecture over layered compatibility shims.
- Remove obsolete interfaces during intended refactors unless backward compatibility is explicitly requested.
- Keep one canonical owner for each type, model, or data contract; legacy paths should be thin wrappers or re-exports, not second implementations.
- Move reusable helper logic into shared utils or clearly named helper modules instead of letting feature modules accumulate generic helper code.
- Use the repo-root `autoimprove.md` as the single ground-truth spec for bounded code-improvement loops; avoid duplicating that prompt/config surface elsewhere.
- In autonomous simplification loops, prefer seams that reduce duplicate contracts, helper collisions, and private helper sprawl before lower-leverage cleanup.
- If an autonomous simplification loop starts stalling, improve the reward or seam-selection signal in `autoimprove.md` before pausing at a status-only report.

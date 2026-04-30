---
id: 2026-04-29_agents_scaffold_omx_integration
date: 2026-04-29
title: "Agents Scaffold OMX Integration"
status: done
topics: [agents, scaffold, omx, skills]
confidence: high
canonical_updates_needed: []
---

Task: implement the AGENTS scaffold cleanup and optional OMX repo-plugin
integration while preserving `.agents/` and canonical memory as the durable
source of truth.

Method: reduced the root `AGENTS.md` to a dispatcher, aligned nested package
and docs guidance with immutable VIN offline stores and hidden internal agent
surfaces, removed stale wrong-repo scaffold examples, added compact
ARIA-specific skills for geometry, docs curation, dataset/offline-store
operations, counterfactual rollouts, and entity-aware RRI, and documented OMX as
optional operator orchestration with checked-in templates only.

Outputs: added `.agents/references/omx_quick_reference.md`,
`.codex/config.example.toml`, `.codex/hooks.example.json`, and
`.codex-plugin/README.md`; ignored `.omx/` and personal `.codex` runtime files;
resolved the completed agents DB records for the thin dispatcher, domain
skills, stale reference audit, and OMX template pilot.

Verification: run `make agents-db AGENTS_ARGS='validate'`, `make agents-db`,
`make check-agent-memory`, local skill validation for each new or changed skill,
the requested stale-reference `rg`, and `git status --short --untracked-files=all`.

Canonical state impact: none. No `.agents/memory/state/` file changed; this was
scaffold, workflow guidance, skill, and backlog maintenance.

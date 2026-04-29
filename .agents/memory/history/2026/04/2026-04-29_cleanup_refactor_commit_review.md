---
id: 2026-04-29_cleanup_refactor_commit_review
date: 2026-04-29
title: "Cleanup Refactor Commit Review"
status: done
topics: [cleanup, data-handling, docs, streamlit, rl, agents]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/GOTCHAS.md
  - .agents/memory/state/OPEN_QUESTIONS.md
  - .agents/memory/state/PROJECT_STATE.md
---

## Task

Review the dirty cleanup/refactor worktree, keep the immutable VIN offline store as the only supported offline training path, do not restore failed migration tooling, and split the reviewed work into self-contained commit packages.

## Method

Reviewed the package, docs, and agent-memory surfaces against root and nested `AGENTS.md` guidance plus the ARIA-NBV code-review skill. Fixed the broken panel dispatcher test, removed stale legacy cache/migration wording from active docs and operator guidance, added immutable VIN offline-store diagnostics, and preserved the deletion of legacy cache and migration modules.

## Findings

The original review findings were valid: the dispatcher test still referenced the deleted `offline_stats` panel, and the slide deck still described `OracleRriCacheDataset` / `VinSnippetCacheProvider` as active. Additional review found stale migration wording in the README, roadmap, slides, diagram sources, and selected runtime messages. Those surfaces now point to `VinOfflineDataset`, `VinOfflineSourceConfig`, and `VinOfflineWriter` where appropriate.

## Verification

Targeted package verification covered panel dispatch, public data-handling contracts, VIN offline-store behavior, RL panel behavior, counterfactual rollout/RL scaffolding, candidate panel behavior, VIN utility behavior, and benchmark plotting helpers. Docs verification covered Mermaid regeneration and Typst compilation for the touched slide and paper sources. Agent-memory verification covered the active memory state and agents DB schema.

## Canonical State Impact

`GOTCHAS.md`, `OPEN_QUESTIONS.md`, and `PROJECT_STATE.md` now reflect the active VIN offline-store path, the removed migration/cache surface, and the new literature/KG routing.

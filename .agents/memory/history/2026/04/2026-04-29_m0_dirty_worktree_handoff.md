---
id: 2026-04-29_m0_dirty_worktree_handoff
date: 2026-04-29
title: "M0 Dirty Worktree Handoff"
status: done
topics: [repo-hygiene, m0, handoff]
confidence: high
canonical_updates_needed: []
artifacts:
  - git status --short
  - git diff --stat -- . ':!docs/_site'
---

## Task

Classify the active dirty worktree before starting larger M1 oracle, data, or
VIN contract work.

## Findings

At handoff time, the visible dirty tree was limited to agent/litkg work:

- `.agents/external/litkg-rs`: modified submodule content in
  `crates/litkg-core/src/semantic_scholar.rs`.
- `.agents/skills/semantic-scholar-litkg/SKILL.md`: skill text updated to
  prefer native litkg-rs Semantic Scholar commands.
- `.agents/memory/history/2026/04/2026-04-29_litkg_semantic_scholar_rest_integration.md`:
  untracked debrief for the litkg Semantic Scholar integration.
- `.agents/issues.toml`: active backlog edits, including priority changes for
  CI/setup issues and new GitHub/scaffold/experiment-registry issues.

No dirty package-code, Streamlit app, RRI, VIN, data-handling, Quarto, Typst,
or config changes were visible in `git status --short` at this checkpoint.

## Boundaries

The litkg and Semantic Scholar changes appear unrelated to the oracle pipeline
implementation slice and should not be staged, reverted, or amended by that
work. Any package-code changes after this note belong to the oracle/M1 slice
unless a later handoff says otherwise.

## Verification

Commands used:

```sh
git status --short
git diff --stat -- . ':!docs/_site'
```

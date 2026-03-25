---
id: 2026-03-24_typst_sync_investigation_2026-03-24
date: 2026-03-24
title: "Typst Sync Investigation 2026 03 24"
status: legacy-imported
topics: [typst, sync, investigation, 2026, 03]
source_legacy_path: ".codex/typst_sync_investigation_2026-03-24.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Typst Sync Investigation (2026-03-24)

## Repository state

- Branch: `main` (tracking `origin/main`)
- Divergence: `ahead 1, behind 3`
- Working tree: very dirty (`217` changed paths, all unstaged)
- Staged changes: none

Remote-only commits:

- `6768196` `chore: clean up`
- `6ab50e1` `Fix Quarto publish workflow for CI`
- `a009de5` `docs: update readme`

Local-only commit:

- `4b60c03` `rm redundant diagrams`

## Typst-specific state

`docs/typst/**` currently has `56` changed paths:

- many modified tracked files in `docs/typst/paper/sections/*.typ`
- modified `docs/typst/paper/main.typ`
- modified `docs/typst/slides/slides_4.typ`
- several tracked deletions under `docs/typst/slides/` (diagram/demo files)
- untracked additions required by current edits, including:
  - `docs/typst/paper/sections/10a-extensions.typ`
  - `docs/typst/slides/data/*.json|*.toml`
  - `docs/typst/paper/data/*.toml|*.csv`

Important: preserving only `main.typ` and `slides_4.typ` is insufficient; both depend on additional new/changed files.

## Recommended recovery flow

1. Create a snapshot branch from current state and commit WIP there.
2. Re-align local `main` to `origin/main`.
3. Create a fresh feature branch from updated `main`.
4. Restore only the Typst-related work from the snapshot commit.
5. Build Typst outputs and commit in small, logical chunks.

## Command sketch

```bash
# 1) Snapshot everything safely
git switch -c codex/wip-local-snapshot-2026-03-24
git add -A
git commit -m "WIP snapshot before main sync (typst + local cleanup)"

# 2) Sync main to remote
git switch main
git fetch origin
git reset --hard origin/main

# 3) Start clean branch for typst work
git switch -c codex/typst-finalize

# 4) Bring back typst scope from snapshot
git checkout codex/wip-local-snapshot-2026-03-24 -- docs/typst

# If needed, also restore bibliography/macros touched by typst edits
git checkout codex/wip-local-snapshot-2026-03-24 -- docs/references.bib docs/typst/shared/macros.typ

# 5) Inspect and commit intentionally
git status -sb
```

## Notes

- The local commit `4b60c03` (diagram removals) is not on `origin/main`. Decide explicitly whether to keep it as a separate PR/branch.
- Use branch-based transfer (`git checkout <branch> -- <path>`) to avoid losing work and avoid broad stashes in this repository state.

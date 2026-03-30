# Agent Memory

This directory replaces the old flat `.codex/*.md` note bucket.

## Layout
- `state/`: canonical current truth that should stay small and current
- `history/YYYY/MM/`: dated task debriefs and imported episodic notes; historical evidence only, not default bootstrap context
- `index/`: migration manifests and machine-oriented indexes

## What Happened To The Old `.codex/*` Debriefs
- Dated or clearly episodic notes were imported into `history/YYYY/MM/` with YAML frontmatter.
- Ambiguous or undated legacy notes were archived under `.agents/archive/codex-legacy/flat/`.
- Previous canonical-input documents such as the old `AGENTS.md` and `AGENTS_INTERNAL_DB.md` were archived under `.agents/archive/codex-legacy/canonical-inputs/`.
- The migration inventory is recorded in `index/codex_migration_manifest.md`.

## Current Policy
- Non-trivial tasks should leave a debrief in `history/YYYY/MM/`.
- Tasks with explicit owner TODOs, issues, core requests, or durable feedback should leave a debrief even when the code or docs change would otherwise be too small for one.
- If a task changes current truth, update one or more files in `state/`.
- Durable owner guidance belongs in `.agents/memory/state/OWNER_DIRECTIVES.md`, not just in dated history notes.
- Native debriefs must include a `## Prompt Follow-Through` section that says whether durable owner prompt items were present and where they were promoted.
- If a task does not change current truth, say so explicitly in the debrief instead of silently relying on chat history.
- Imported or older history entries may reference stale repo paths from previous layouts; treat them as historical evidence until current state docs prove otherwise.

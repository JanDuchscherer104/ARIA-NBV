# Agent Memory

This directory replaces the old flat `.codex/*.md` note bucket.

## Layout
- `state/`: canonical current truth that should stay small and current
- `history/YYYY/MM/`: dated task debriefs and imported episodic notes
- `index/`: migration manifests and machine-oriented indexes

## What Happened To The Old `.codex/*` Debriefs
- Dated or clearly episodic notes were imported into `history/YYYY/MM/` with YAML frontmatter.
- Ambiguous or undated legacy notes were archived under `archive/codex-legacy/flat/`.
- Previous canonical-input documents such as the old `AGENTS.md` and `AGENTS_INTERNAL_DB.md` were archived under `archive/codex-legacy/canonical-inputs/`.
- The migration inventory is recorded in `index/codex_migration_manifest.md`.

## Current Policy
- Non-trivial tasks should leave a debrief in `history/YYYY/MM/`.
- If a task changes current truth, update one or more files in `state/`.
- If a task does not change current truth, say so explicitly in the debrief instead of silently relying on chat history.

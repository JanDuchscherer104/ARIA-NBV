# ARIA-NBV Agent Guidance

Use this file as the root dispatcher. Detailed rules live in the nearest
`AGENTS.md`, `.agents/skills/`, and `.agents/references/`.

## Source Order
- High-level project truth: `docs/typst/seminar_paper/main.typ`.
- Current durable state: `.agents/memory/state/PROJECT_STATE.md`,
  `DECISIONS.md`, `OPEN_QUESTIONS.md`, and `GOTCHAS.md`.
- Active backlog: `.agents/issues.toml`, `.agents/todos.toml`,
  `.agents/refactors.toml`, and `.agents/resolved.toml` via `make agents-db`.
- Lightweight generated routing: `docs/_generated/context/source_index.md`,
  `literature_index.md`, and `data_contracts.md`; refresh with `make context`
  when stale.
- Operator aids and long conventions: `.agents/references/`.
- Durable human-owner preferences:
  `.agents/references/human_owner_intent.md`.

## Routing
- Package work under `aria_nbv/`: read `aria_nbv/AGENTS.md`, then the nested
  guide for `data_handling`, `rri_metrics`, or `vin` when that contract is
  touched.
- Docs, bibliography, Typst, or Quarto work: read `docs/AGENTS.md`.
- Broad cross-surface questions: use the `aria-nbv-context` skill first.
- Vague, high-impact, or advisor-facing plans: use `plan-grill`.
- Bugs, regressions, suspicious metrics, or failing docs/data/KG checks: use
  `diagnose-aria`.
- Backlog or memory changes: use the `agents-db` skill.
- Cleanup, pruning, or simplification: use the `simplification` skill.
- LRZ AI Systems, Slurm, DSS, Pyxis, or remote compute work: use `lrz-ai-systems`.
- KG/literature/code graph work: use `semantic-scholar-litkg`; keep
  repo-independent implementation in `.agents/external/litkg-rs`.
- OMX is optional operator orchestration. Use
  `.agents/references/omx_quick_reference.md`; do not make OMX required for
  normal repo work.

## Non-Negotiables
- Do not use `git restore` or `git reset --hard` unless explicitly requested.
- Assume the worktree can be dirty; never revert unrelated user or agent
  changes.
- Keep public docs aligned with the Typst paper and current code.
- Internal agent memory, generated context, and OMX runtime state are not public
  documentation surfaces.

## Instruction Capture
- Repo invariant: update this file or the nearest nested `AGENTS.md`.
- Repeatable workflow: update or add a compact `.agents/skills/*/SKILL.md`.
- Human-owner preference: update `.agents/references/human_owner_intent.md`.
- Current truth: update `.agents/memory/state/`.
- Actionable work: update `.agents/issues.toml`, `.agents/todos.toml`, or
  `.agents/refactors.toml` through `agents-db`.
- Public narrative: update Quarto or Typst docs.

## Commands
- Python: `aria_nbv/.venv/bin/python`
- Environment recovery: `cd aria_nbv && uv sync --all-extras`
- Package format/lint: `ruff format <file>` and `ruff check <file>`
- Package tests: `cd aria_nbv && uv run pytest <path>`
- Context refresh: `make context`
- Contract surface: `make context-contracts`
- KG profile: `.configs/litkg.toml`; use `make kg-sync`,
  `make kg-materialize`, `make kg-semantic-enrich`, `make kg-export-neo4j`,
  and `make kg-index-code` for the ARIA-NBV litkg-rs integration.
- Agent memory check: `make check-agent-memory`
- Agents DB: `make agents-db` or `make agents-db AGENTS_ARGS='validate'`

## Verification
- Repo guidance, canonical state, debriefs, or skills:
  `make check-agent-memory`; validate changed skills with the local skill
  validator when available.
- Agents DB edits: `make agents-db AGENTS_ARGS='validate'` and `make agents-db`.
- Python/package edits: format, lint, and targeted pytest for the touched
  surface.
- Data-handling, RRI, or VIN contract edits: follow the nearest nested guide and
  update docs/memory when behavior changes.
- Docs edits: render the touched Quarto or Typst surface when non-trivial.

## Debriefs
- Non-trivial work leaves a debrief under `.agents/memory/history/YYYY/MM/`.
- Native debriefs must follow `.agents/references/agent_memory_templates.md` and
  include `canonical_updates_needed` even when the list is empty.
- Legacy `.codex/*.md` notes were migrated. Do not recreate `.codex` as a notes
  bucket; only checked-in `.codex/*.example.*` templates are allowed.

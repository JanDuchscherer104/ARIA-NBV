# NBV Codex Guidance

Use the documented Codex repo surfaces in this repository:
- repo guidance lives in this file and nested `AGENTS.md` files
- repo skills live in `.agents/skills/`
- project memory lives in `.agents/memory/`
- agent-facing reference docs live in `.agents/references/`
- generated context lives in `docs/_generated/context/`

## Start Here
- Refresh lightweight routing artifacts with `make context` when the scaffold may be stale.
- Default bootstrap: [docs/typst/paper/main.typ](/home/jandu/repos/NBV/docs/typst/paper/main.typ), the canonical state docs in `.agents/memory/state/`, and `docs/_generated/context/source_index.md`.
- If `aria_nbv/.venv` is missing or stale, rebuild it with `UV_PYTHON=/home/jandu/miniforge3/envs/aria-nbv/bin/python uv sync --all-extras`.
- Use `make context-contracts` or `scripts/nbv_get_context.sh contracts` before falling back to `docs/_generated/context/data_contracts.md`.
- Use [docs/index.qmd](/home/jandu/repos/NBV/docs/index.qmd) and [docs/contents/todos.qmd](/home/jandu/repos/NBV/docs/contents/todos.qmd) only when the task is about project narrative, roadmap, or active work items.
- Use `make context-heavy` or the specific `context-uml`, `context-docstrings`, `context-tree`, and `context-dir-tree` targets only for explicit heavy fallback work.
- Use [operator_quick_reference.md](/home/jandu/repos/NBV/.agents/references/operator_quick_reference.md) when you need environment recovery, repo hygiene, or compact ASE/EFM quick references.
- Prefer the `aria-nbv-context` skill for targeted cross-doc retrieval instead of loading large static dumps.

## Repo Map
- `aria_nbv/`: package code, configs, CLIs, and tests.
- `docs/`: Quarto docs, Typst paper/slides, and bibliography.
- `.agents/memory/`: canonical state, gotchas, epis debriefs, and migration indexes.
- `.agents/references/`: agent-readable reference docs such as Python conventions and Context7 IDs.
- `scripts/`: repo-level helper wrappers and migration utilities.
- `tests/` and `aria_nbv/tests/`: verification.

## Commands
- Python: `aria_nbv/.venv/bin/python`
- Environment recovery: `UV_PYTHON=/home/jandu/miniforge3/envs/aria-nbv/bin/python uv sync --all-extras`
- Format: `ruff format <file>`
- Lint: `ruff check <file>`
- Tests: `uv run pytest <path>`
- Context refresh: `make context`
- Contracts surface: `make context-contracts`
- Heavy context fallback: `make context-heavy`
- Memory hygiene: `make check-agent-memory`

## Agentic Behaviors
- Read `docs/typst/paper/main.typ` first for the top-level project narrative.
- Start by condensing the task, defining acceptance criteria, and identifying the exact files or symbols involved.
- Treat explicit user termination criteria as binding; expand verification until they are satisfied or explain precisely what remains blocked.
- Inspect referenced files and symbols before editing; prefer `rg`, targeted file reads, and the repo skill over broad dumps.
- Maintain a small task list for multi-step work and verify incrementally instead of batching untested changes.
- Assume the environment works unless the user says otherwise, but confirm the exact interpreter and command path before diagnosing dependency problems.
- Stay within the requested scope. If something adjacent looks wrong but was not requested, note it instead of silently changing it.
- Keep documentation aligned with behavior changes.
- Summarize important findings and leave a debrief for non-trivial work.

## Non-Negotiable Constraints
- Do not use `git restore` or `git reset --hard` unless the user explicitly asks.
- Assume the worktree is dirty; work around unrelated changes instead of reverting them.
- Keep pose and camera types as `PoseTW` and `CameraTW`.
- Follow the config-as-factory pattern via `.setup_target()`.
- Treat `docs/typst/paper/main.typ` as the authoritative high-level description when Quarto docs disagree.
- Update docs when code changes alter behavior or user-facing workflows.

## Done Means
- Changed files are formatted or linted as appropriate.
- Targeted tests or verification commands were run for the changed surface.
- Real-data or integration-style checks were used when feasible for package behavior changes.
- No temporary placeholders or stale path references remain.
- For scaffold or debrief changes, `make check-agent-memory` passes.
- If project truth changed, update the relevant files in `.agents/memory/state/`.

## Debriefs
- For non-trivial work, write a debrief record under `.agents/memory/history/YYYY/MM/`.
- Native debriefs must follow `.agents/references/agent_memory_templates.md`.
- Native debriefs must include `canonical_updates_needed: []` when no canonical state doc changed.
- Existing `status: legacy-imported` records are archive evidence; do not backfill them unless a task explicitly requires it.
- If the task changes current truth, update one or more files in `.agents/memory/state/`; otherwise record `canonical_updates_needed: []`.
- Legacy `.codex` notes were migrated into `.agents/memory/history/` and `archive/codex-legacy/`. Do not recreate `.codex` as a task-notes bucket.

## Key References
- [source_index.md](/home/jandu/repos/NBV/docs/_generated/context/source_index.md)
- [operator_quick_reference.md](/home/jandu/repos/NBV/.agents/references/operator_quick_reference.md)
- [python_conventions.md](/home/jandu/repos/NBV/.agents/references/python_conventions.md)
- [context7_library_ids.md](/home/jandu/repos/NBV/.agents/references/context7_library_ids.md)
- [agent_memory_templates.md](/home/jandu/repos/NBV/.agents/references/agent_memory_templates.md)
- [GOTCHAS.md](/home/jandu/repos/NBV/.agents/memory/state/GOTCHAS.md)
- `notebooks/ase_oracle_rri_simplified.ipynb`

## Retrieve On Demand
- [docs/index.qmd](/home/jandu/repos/NBV/docs/index.qmd)
- [docs/contents/todos.qmd](/home/jandu/repos/NBV/docs/contents/todos.qmd)
- [docs/contents/roadmap.qmd](/home/jandu/repos/NBV/docs/contents/roadmap.qmd)
- [docs/contents/questions.qmd](/home/jandu/repos/NBV/docs/contents/questions.qmd)
- `docs/contents/impl/`
- `docs/contents/ext-impl/`

## Recurring Gotchas
- [GOTCHAS.md](/home/jandu/repos/NBV/.agents/memory/state/GOTCHAS.md) is the canonical maintained gotcha list.
- Validation is disabled by default unless `trainer_config.enable_validation=true`.
- Use `Field(default_factory=...)` for computed defaults; do not store callables in `Field(default=...)`.
- Prefer `uv run pytest` or the repo venv python over the system interpreter.
- Offline cache splits are file-backed and may create or update `train_index.jsonl` and `val_index.jsonl`.

## Canonical State
- [Project State](/home/jandu/repos/NBV/.agents/memory/state/PROJECT_STATE.md)
- [Decisions](/home/jandu/repos/NBV/.agents/memory/state/DECISIONS.md)
- [Open Questions](/home/jandu/repos/NBV/.agents/memory/state/OPEN_QUESTIONS.md)
- [Gotchas](/home/jandu/repos/NBV/.agents/memory/state/GOTCHAS.md)

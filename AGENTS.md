# NBV Codex Guidance

Use the documented Codex repo surfaces in this repository:
- repo guidance lives in this file and nested `AGENTS.md` files
- repo skills live in `.agents/skills/`
- project memory lives in `.agents/memory/`
- agent-facing reference docs live in `.agents/references/`
- generated context lives in `docs/_generated/context/`

## Start Here
- Refresh lightweight routing artifacts with `make context` when the scaffold may be stale.
- Default bootstrap: [docs/typst/paper/main.typ](docs/typst/paper/main.typ), the canonical state docs in `.agents/memory/state/`, and `docs/_generated/context/source_index.md`.
- If `aria_nbv/.venv` is missing or stale, rebuild it from `aria_nbv/` with `uv sync --all-extras`. If `uv` needs an explicit interpreter on this machine, set `UV_PYTHON` to a local Python 3.11 path and keep that host-specific value in operator aids rather than shared repo guidance.
- Use `make context-contracts` or `scripts/nbv_get_context.sh contracts` before falling back to `docs/_generated/context/data_contracts.md`.
- Use [docs/index.qmd](docs/index.qmd) and [docs/contents/todos.qmd](docs/contents/todos.qmd) only when the task is about project narrative, roadmap, or active work items.
- Use `make context-heavy` or the specific `context-uml`, `context-docstrings`, `context-tree`, and `context-dir-tree` targets only for explicit heavy fallback work.
- Use [operator_quick_reference.md](.agents/references/operator_quick_reference.md) when you need environment recovery, repo hygiene, or compact ASE/EFM quick references.
- Prefer the `aria-nbv-context` skill for targeted cross-doc retrieval instead of loading large static dumps.

## Progressive Disclosure
- Stay on the hot path until the task localizes: [docs/typst/paper/main.typ](docs/typst/paper/main.typ), `.agents/memory/state/`, and `docs/_generated/context/source_index.md`.
- Open one deeper guide only after the touched surface is clear:
  - [aria_nbv/AGENTS.md](aria_nbv/AGENTS.md) for package code, configs, CLIs, tests, and Python workflow under `aria_nbv/`
  - [docs/AGENTS.md](docs/AGENTS.md) for Quarto, Typst, bibliography, and render workflow changes under `docs/`
  - [aria_nbv/aria_nbv/data_handling/AGENTS.md](aria_nbv/aria_nbv/data_handling/AGENTS.md) for snippet, cache, dataset, and contract changes
  - [aria_nbv/aria_nbv/rri_metrics/AGENTS.md](aria_nbv/aria_nbv/rri_metrics/AGENTS.md) for oracle RRI, binning, and metric semantics
  - [aria_nbv/aria_nbv/vin/AGENTS.md](aria_nbv/aria_nbv/vin/AGENTS.md) for VIN scorer, batch-contract, and candidate-context changes
- If the task crosses surfaces, start with the owner of the main contract, then open adjacent guides only for the affected boundary.
- Prefer the `aria-nbv-context` skill to localize symbols, modules, or document sections before opening broader generated context.
- Use `make context-contracts` / `scripts/nbv_get_context.sh contracts` before broader generated artifacts; use `make context-heavy` only as an explicit fallback.

## Tech Stack
- Python 3.11 package in `aria_nbv/`, managed with `uv`.
- ML and geometry core: PyTorch, PyTorch3D, PyTorch Lightning, NumPy, SciPy, Pydantic, Open3D, trimesh, zarr, Optuna, Weight and Biases.
- Aria / EFM stack: `projectaria-tools`, `projectaria-atek`, and local `efm3d`.
- Quality and verification: Ruff, MyPy, and Pytest.

## Repo Map
- `aria_nbv/`: package code, configs, CLIs, and tests.
- `docs/`: Quarto docs, Typst paper/slides, and bibliography.
- `.agents/memory/`: canonical state, gotchas, epis debriefs, and migration indexes.
- `.agents/references/`: agent-readable reference docs such as Python conventions and Context7 IDs.
- `scripts/`: repo-level helper wrappers, context entrypoints, and migration utilities. Keep reusable repository entrypoints here; place agent-only implementation details under `.agents/`.
- `tests/` and `aria_nbv/tests/`: verification.

## Commands
- Python: `aria_nbv/.venv/bin/python`
- Environment recovery: `cd aria_nbv && uv sync --all-extras`
- Explicit interpreter override when needed: `cd aria_nbv && UV_PYTHON=<local-python-3.11-path> uv sync --all-extras`
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
- Inspect referenced and potentially affected files and modules before editing.
- Maintain a small task list for multi-step work and verify incrementally instead of batching untested changes.
- Stay within the requested scope. If something adjacent looks wrong but was not requested, note it instead of silently changing it.
- Keep documentation aligned with behavior changes.
- Summarize important findings and leave a debrief for non-trivial work. If the scope of the work appears too broad and criteria, constraints, or verification steps are unclear, clarify them with the user before proceeding.

## Non-Negotiable Constraints
- Do not use `git restore` or `git reset --hard` unless the user explicitly asks.
- Assume the worktree is dirty; work around unrelated changes instead of reverting them.
- Keep pose and camera types as `PoseTW` and `CameraTW`.
- Follow the config-as-factory pattern via `.setup_target()`.
- Treat `docs/typst/paper/main.typ` as the authoritative high-level description when Quarto docs disagree.
- Update docs when code changes alter behavior or user-facing workflows.
- Do *not* reinvent the wheel; prefer existing implementations and utilities from public libraries and our interal codebase. Before implementing new features try to identify highly suitable existing implementations or packages than can be adapted to our needs. Then inspect API docs (**context7**) or source code to confirm suitability and understand how to use them effectively for the task.


## Verification Matrix
- Repo guidance, canonical state, or debrief changes: run `make check-agent-memory`.
- Context scaffolding or contract-routing changes: run the most direct target among `make context`, `make context-contracts`, or the specific `scripts/nbv_get_context.sh` mode that was touched.
- Python/package changes under `aria_nbv/`: run `ruff format <file>`, `ruff check <file>`, and targeted `uv run pytest <path>`.
- Data-handling, RRI, or VIN contract changes: follow the matching nested guide and run its targeted tests plus any required docs update.
- Docs-only changes under `docs/`: run the most direct render/check step when the change is non-trivial, affects build behavior, or changes published output.
- Behavior or workflow changes visible across surfaces: update docs and `.agents/memory/state/` in the same change.

## Completion Criteria
- The verification row matching the touched surface was satisfied.
- No temporary placeholders or stale path references remain.
- Docs and canonical state reflect any changed behavior or workflow.
- Non-trivial work leaves a debrief under `.agents/memory/history/YYYY/MM/`.

## Debriefs
- For non-trivial work, write a debrief record under `.agents/memory/history/YYYY/MM/`.
- Native debriefs must follow `.agents/references/agent_memory_templates.md`.
- Native debriefs must include `canonical_updates_needed: []` when no canonical state doc changed.
- Existing `status: legacy-imported` records are archive evidence; do not backfill them unless a task explicitly requires it.
- If the task changes current truth, update one or more files in `.agents/memory/state/`; otherwise record `canonical_updates_needed: []`.
- Legacy `.codex` notes were migrated into `.agents/memory/history/` and `archive/codex-legacy/`. Do not recreate `.codex` as a task-notes bucket.

## Key References
- [source_index.md](docs/_generated/context/source_index.md)
- [operator_quick_reference.md](.agents/references/operator_quick_reference.md)
- [python_conventions.md](.agents/references/python_conventions.md)
- [context7_library_ids.md](.agents/references/context7_library_ids.md)
- [agent_memory_templates.md](.agents/references/agent_memory_templates.md)
- [GOTCHAS.md](.agents/memory/state/GOTCHAS.md)
- `notebooks/ase_oracle_rri_simplified.ipynb`

## Retrieve On Demand
- [docs/index.qmd](docs/index.qmd)
- [docs/contents/ideas.qmd](docs/contents/ideas.qmd)
- [docs/contents/todos.qmd](docs/contents/todos.qmd)
- [docs/contents/roadmap.qmd](docs/contents/roadmap.qmd)
- [docs/contents/questions.qmd](docs/contents/questions.qmd)
- `docs/contents/impl/`
- `docs/contents/ext-impl/`

## Recurring Gotchas
- [GOTCHAS.md](.agents/memory/state/GOTCHAS.md) is the canonical maintained gotcha list.
- Validation is disabled by default unless `trainer_config.enable_validation=true`.
- Use `Field(default_factory=...)` for computed defaults; do not store callables in `Field(default=...)`.
- Prefer `uv run pytest` or the repo venv python over the system interpreter.
- Offline cache splits are file-backed and may create or update `train_index.jsonl` and `val_index.jsonl`.

## Canonical State
- [Project State](.agents/memory/state/PROJECT_STATE.md)
- [Decisions](.agents/memory/state/DECISIONS.md)
- [Open Questions](.agents/memory/state/OPEN_QUESTIONS.md)
- [Gotchas](.agents/memory/state/GOTCHAS.md)

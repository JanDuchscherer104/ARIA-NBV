# NBV Codex Guidance

Use the documented Codex repo surfaces in this repository:
- repo-wide policy lives in this file and nested `AGENTS.md` files
- canonical project state lives in `.agents/memory/state/`
- repo skills live in `.agents/skills/`
- agent-facing references live in `.agents/references/`
- generated context lives in `docs/_generated/context/`

## Sources of Truth
- Treat [docs/typst/paper/main.typ](docs/typst/paper/main.typ) as the highest-level project narrative.
- Treat the canonical state docs in `.agents/memory/state/` as current truth:
  - [PROJECT_STATE.md](.agents/memory/state/PROJECT_STATE.md)
  - [DECISIONS.md](.agents/memory/state/DECISIONS.md)
  - [OPEN_QUESTIONS.md](.agents/memory/state/OPEN_QUESTIONS.md)
  - [GOTCHAS.md](.agents/memory/state/GOTCHAS.md)
- Treat `docs/_generated/context/` as derived routing output, not canonical state.

## Hierarchy Rules
- Stay on the hot path until the task localizes: [docs/typst/paper/main.typ](docs/typst/paper/main.typ), `.agents/memory/state/`, and the compact `docs/_generated/context/source_index.md` when it has been generated.
- Open one deeper guide only after the touched surface is clear:
  - [aria_nbv/AGENTS.md](aria_nbv/AGENTS.md) for package code, configs, CLIs, tests, and Python workflow under `aria_nbv/`
  - [docs/AGENTS.md](docs/AGENTS.md) for Quarto, Typst, bibliography, and render workflow changes under `docs/`
  - [aria_nbv/aria_nbv/data_handling/AGENTS.md](aria_nbv/aria_nbv/data_handling/AGENTS.md) for snippet, cache, dataset, and contract changes
  - [aria_nbv/aria_nbv/rri_metrics/AGENTS.md](aria_nbv/aria_nbv/rri_metrics/AGENTS.md) for oracle RRI, binning, and metric semantics
  - [aria_nbv/aria_nbv/vin/AGENTS.md](aria_nbv/aria_nbv/vin/AGENTS.md) for VIN scorer, batch-contract, and candidate-context changes
- If the task crosses surfaces, start with the owner of the main contract, then open adjacent guides only for the affected boundary.
- Prefer the `aria-nbv-context` skill before broad generated dumps or wide file reads.

## Detailed Owners
- Bootstrap, routing, context refresh, contract routing, and on-demand retrieval entrypoints: [.agents/skills/aria-nbv-context/SKILL.md](.agents/skills/aria-nbv-context/SKILL.md)
- Environment recovery, repo hygiene, and repo-wide execution hygiene: [.agents/references/operator_quick_reference.md](.agents/references/operator_quick_reference.md)
- Package commands, Python workflow, and package-level verification: [aria_nbv/AGENTS.md](aria_nbv/AGENTS.md)
- Docs commands, render workflow, and documentation-specific rules: [docs/AGENTS.md](docs/AGENTS.md)
- Tech stack and stable repo conventions: [.agents/memory/state/PROJECT_STATE.md](.agents/memory/state/PROJECT_STATE.md)
- Durable scaffold decisions: [.agents/memory/state/DECISIONS.md](.agents/memory/state/DECISIONS.md)
- Recurring pitfalls: [.agents/memory/state/GOTCHAS.md](.agents/memory/state/GOTCHAS.md)
- Debrief templates and memory-record rules: [.agents/references/agent_memory_templates.md](.agents/references/agent_memory_templates.md)
- External library lookup IDs: [.agents/references/context7_library_ids.md](.agents/references/context7_library_ids.md)

## Repo Map
- `aria_nbv/`: package code, configs, CLIs, and tests.
- `docs/`: Quarto docs, Typst paper/slides, and bibliography.
- `.agents/memory/`: canonical state, gotchas, episodic debriefs, and migration indexes.
- `.agents/references/`: agent-readable reference docs such as Python conventions and Context7 IDs.
- `scripts/`: repo-level helper wrappers, context entrypoints, and migration utilities. Keep reusable repository entrypoints here; place agent-only implementation details under `.agents/`.
- `tests/` and `aria_nbv/tests/`: verification.

## Non-Negotiable Constraints
- Do not use `git restore` or `git reset --hard` unless the user explicitly asks.
- Assume the worktree is dirty; work around unrelated changes instead of reverting them.
- Keep pose and camera types as `PoseTW` and `CameraTW`.
- Follow the config-as-factory pattern via `.setup_target()`.
- Treat `docs/typst/paper/main.typ` as the authoritative high-level description when Quarto docs disagree.
- Update docs when code changes alter behavior or user-facing workflows.
- Do *not* reinvent the wheel; prefer existing implementations and utilities from public libraries and our internal codebase. Before implementing new features try to identify highly suitable existing implementations or packages that can be adapted to our needs. Then inspect API docs (**context7**) or source code to confirm suitability and understand how to use them effectively for the task.

## High-Level Verification
- Repo guidance, canonical state, or debrief changes: run `make check-agent-memory`.
- Context scaffolding or contract-routing changes: run the most direct target among `make context`, `make context-contracts`, or the specific `scripts/nbv_get_context.sh` mode that was touched.
- Python/package changes under `aria_nbv/`: follow [aria_nbv/AGENTS.md](aria_nbv/AGENTS.md) and run its format/lint/pytest stack.
- Data-handling, RRI, or VIN contract changes: follow the matching nested guide and run its targeted tests plus any required docs update.
- Docs-only changes under `docs/`: follow [docs/AGENTS.md](docs/AGENTS.md) and run the most direct render/check step when required.
- Behavior or workflow changes visible across surfaces: update docs and `.agents/memory/state/` in the same change.

## Completion Criteria
- The verification row matching the touched surface was satisfied.
- No temporary placeholders or stale path references remain.
- Docs and canonical state reflect any changed behavior or workflow.
- Non-trivial work leaves a debrief under `.agents/memory/history/YYYY/MM/`.

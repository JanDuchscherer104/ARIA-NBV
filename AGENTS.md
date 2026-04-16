# NBV Agent Guidance

This repository develops Aria-NBV: an active next-best-view research stack for
egocentric indoor reconstruction, RRI oracle supervision, VIN-style candidate
scoring.

## Sources Of Truth
- `AGENTS.md`: repo-wide agent policy and routing only. Nested `AGENTS.md`
  files add subtree-specific deltas.
- `docs/typst/paper/main.typ`: highest-level project narrative when docs differ.
- `.agents/memory/state/`: canonical current truth, decisions, open questions,
  and maintained gotchas.
- `docs/_generated/context/source_index.md`: generated lightweight routing index.
- `.agents/AGENTS_INTERNAL_DB.md`: compact agent/tooling alignment facts.
- `.agents/references/`: long-form conventions, operator aids, and optional tool
  references.
- The nearest nested `AGENTS.md` overrides this file for its subtree.

## Start Here
- New trusted Codex sessions run the repo startup hook, which refreshes the
  lightweight generated context bundle under `docs/_generated/context/`.
- Default bootstrap is `docs/typst/paper/main.typ`,
  `.agents/memory/state/PROJECT_STATE.md`, and the refreshed generated context:
  `docs/_generated/context/source_index.md`,
  `docs/_generated/context/literature_index.md`, and
  `docs/_generated/context/data_contracts.md`.
- Run `make context` manually only after changing routing or scaffold inputs,
  when hooks are disabled, or when the generated context appears stale.
- Use `aria_nbv/AGENTS.md` for Python package work and `docs/AGENTS.md` for
  documentation work.
- Use a narrower repo skill when discovery is not already localized:
  `aria-nbv-context`, `aria-nbv-docs-context`, `aria-nbv-code-context`, or
  `aria-nbv-scaffold-maintenance`.

## Repo Map
- `aria_nbv/`: Python package workspace, configs, CLIs, and tests.
- `docs/`: Quarto docs, Typst paper/slides, figures, and bibliography.
- `.agents/`: repo-local skills, memory, references, active agent DB, and
  inactive guardrail hook templates.
- `.codex/`: trusted-project Codex config and startup hook that refreshes
  lightweight context for new sessions.
- `scripts/`: stable repo-level helper wrappers and validators.
- `docs/_generated/context/`: generated routing artifacts; refresh instead of
  editing by hand.

## Nested Guides
- `aria_nbv/AGENTS.md`: package-wide Python rules and deeper package routing.
- `aria_nbv/aria_nbv/app/AGENTS.md`: Streamlit app and panel guidance.
- `aria_nbv/aria_nbv/configs/AGENTS.md`: config ownership and path/W&B/Optuna
  config guidance.
- `aria_nbv/aria_nbv/data_handling/AGENTS.md`: raw snippet, cache, and dataset
  contracts.
- `aria_nbv/aria_nbv/lightning/AGENTS.md`: trainer, datamodule, and experiment
  orchestration.
- `aria_nbv/aria_nbv/pipelines/AGENTS.md`: pipeline entrypoint and oracle-labeler
  workflow guidance.
- `aria_nbv/aria_nbv/pose_generation/AGENTS.md`: candidate sampling, feasibility,
  and counterfactual pose helpers.
- `aria_nbv/aria_nbv/rendering/AGENTS.md`: depth rendering, point clouds, and
  rendering diagnostics.
- `aria_nbv/aria_nbv/rl/AGENTS.md`: discrete-shell RL and counterfactual
  environment guidance.
- `aria_nbv/aria_nbv/rri_metrics/AGENTS.md`: oracle RRI, binning, and metric
  semantics.
- `aria_nbv/aria_nbv/vin/AGENTS.md`: VIN scorer, batch contract, and candidate
  context guidance.
- `docs/AGENTS.md`: Quarto, Typst, bibliography, and render workflows.
- `docs/typst/paper/AGENTS.md`: paper-specific Typst and manuscript guidance.

## Repo-Wide Rules
- Stay within the requested scope; note adjacent problems instead of silently
  expanding work.
- Assume the worktree is dirty and preserve unrelated user changes.
- Do not use destructive git commands such as `git restore`, `git reset --hard`,
  or broad checkout/reset operations unless explicitly requested.
- Keep shared guidance portable. Host-specific interpreter paths, caches, and
  operator recovery notes belong in `.agents/references/` or user-local notes.
- Prefer existing implementations and libraries before adding new local
  infrastructure. When external API details matter, inspect official docs or
  vendored source before implementing.
- Keep pose and camera types as `PoseTW` and `CameraTW`; follow config-as-factory
  construction through `.setup_target()`.
- Update docs and canonical memory when behavior or workflow truth changes.

## Verification
- Agent scaffold, guidance, skills, hooks, or agent DB changes: run
  `make check-agent-scaffold`.
- Canonical memory or debrief-only changes: run `make check-agent-memory`.
- Context routing changes: run `make context` and the specific touched context
  target after edits when applicable.
- Python/package changes under `aria_nbv/`: run `ruff format <file>`,
  `ruff check <file>`, and targeted `uv run pytest <path>` from the package
  workspace.
- Docs-only changes under `docs/`: run the most direct render/check target when
  the change affects published output or build behavior.

## Agent Memory And DB
- Current truth lives in `.agents/memory/state/`; episodic debriefs live in
  `.agents/memory/history/YYYY/MM/`.
- Non-trivial work should leave a native debrief following
  `.agents/references/agent_memory_templates.md`.
- Agent/tooling debt is tracked in `.agents/issues.toml`, `.agents/todos.toml`,
  and `.agents/resolved.toml`; inspect it with `make agents-db`.
- Do not recreate `.codex` as a task-notes bucket. The tracked `.codex` surface
  is reserved for repo-local Codex config and startup hooks. Inactive guardrail
  hook templates live under `.agents/references/codex_hook_templates/`.

## Optional Tools
- GitNexus is optional in this Codex scaffold. If available, use the optional
  workflow in `.agents/references/gitnexus_optional.md`; if unavailable, use
  normal targeted code search, local impact inspection, and tests.

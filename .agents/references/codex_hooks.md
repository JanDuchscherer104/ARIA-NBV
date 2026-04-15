# Codex Hooks

This repo tracks an active startup hook plus inactive guardrail templates.

Current Codex hook constraints from the official docs:

- Hooks require `[features] codex_hooks = true` in a Codex config layer.
- Codex discovers active `hooks.json` next to active config layers, including
  `<repo>/.codex/hooks.json`.
- `PreToolUse` and `PostToolUse` currently intercept Bash only.
- Hooks are experimental and should be treated as guardrails, not complete
  policy enforcement.

## Active Startup Hook

- `.codex/config.toml` enables Codex hooks for trusted project sessions.
- `.codex/hooks.json` registers a `SessionStart` hook for `startup` only.
- `.codex/hooks/run_make_context_clean.sh` runs `make context`, strips ANSI
  color codes, and prints an agent-facing summary of the generated
  `docs/_generated/context/` files.

The startup hook refreshes:

- `source_index.md`: routing map, hot-path files, local `AGENTS.md` inventory,
  reveal commands, and search recipes.
- `literature_index.md`: checked-in literature families and source-search
  entrypoints.
- `data_contracts.md`: AST-derived data/config contracts and package-boundary
  objects.

## Inactive Templates

Guardrail hook examples live in `.agents/references/codex_hook_templates/` so
they are versioned but not auto-loaded. Copy or adapt them into `.codex` only
after validating noise level and failure behavior.

# AGENTS Internal Database

Purpose: compact, repository-local alignment memory for agent-scaffold facts,
tooling boundaries, and current maintenance context. This is operational memory
for agents, not a replacement for root `AGENTS.md`, nested guides, or durable
package documentation.

## Mission Snapshot

- Keep the NBV agent scaffold thin, portable, and discoverable.
- Route work through root and nested `AGENTS.md` files, self-contained skills,
  canonical memory, and generated lightweight context.
- Track active agent/tooling debt in `.agents/issues.toml` and
  `.agents/todos.toml`; move completed records to `.agents/resolved.toml`.

## Stable Scaffold Facts

- Root `AGENTS.md` is repo-wide policy and routing only.
- Nested `AGENTS.md` files are delta guidance for their subtree.
- Durable subsystem ownership belongs in local README/REQUIREMENTS files when
  that prevents `AGENTS.md` bloat.
- GitNexus is optional; mandatory GitNexus policy belongs outside root guidance.
- `.codex/hooks.json` owns the active startup hook that runs `make context` for
  new Codex sessions.
- Guardrail hook examples remain inactive templates under
  `.agents/references/codex_hook_templates/`.
- Published agent-scaffold docs are generated from canonical markdown by
  `scripts/quarto_generate_agent_docs.py`.

## DB Scope

Use the DB for:

- stale or contradictory agent guidance
- broken scaffold commands or validators
- skill routing drift
- hook-template and optional-tool workflow debt
- agent-scaffold publication gaps

Do not use the DB as the general research, product, or code backlog.

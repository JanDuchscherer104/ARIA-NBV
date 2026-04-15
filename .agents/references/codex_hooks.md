# Codex Hook Templates

This repo keeps Codex hook examples inactive by default. The templates live in
`.agents/references/codex_hook_templates/` so Codex will not auto-load them as
repo-local hooks.

Current Codex hook constraints from the official docs:

- Hooks require `[features] codex_hooks = true` in a Codex config layer.
- Codex discovers active `hooks.json` next to active config layers, including
  `<repo>/.codex/hooks.json`.
- `PreToolUse` and `PostToolUse` currently intercept Bash only.
- Hooks are experimental and should be treated as guardrails, not complete
  policy enforcement.

To activate later, copy the template JSON into `.codex/hooks.json`, update paths
to use the git-root pattern shown in the template, and enable hooks in a trusted
Codex config layer.

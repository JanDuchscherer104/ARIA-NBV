# Tooling And Skill Governance

Use this reference when adding or changing repo skills, command wrappers, or agent-facing workflow guidance.

## Tooling Policy
- Prefer repo-owned wrappers, pinned local tools, or preinstalled commands in `AGENTS.md` and `SKILL.md`.
- Do not use runtime network fetches in scaffolded workflows. Avoid one-shot package installers and ad-hoc HTTP fetch helpers in agent guidance.
- If a tool must be installed separately, document the local command the agent should call after installation, not the installation command itself.

## Skill Invocation Policy
- Implicit invocation is acceptable for read-only routing or localization skills that do not edit files, call external services, or perform destructive actions.
- Disable or avoid implicit invocation for skills that edit code, call external/networked tools, or trigger high-cost or high-risk operations.
- Keep each skill focused on one job with clear trigger boundaries and a clear handoff or stop rule.

## Review Expectations
- Treat changes to `AGENTS.md`, `SKILL.md`, and scaffold helper scripts as code-reviewable infrastructure.
- Update scaffold validation when adding new hot-path instruction files, wrappers, or governance rules.
- Keep governance details on demand in `.agents/references/`; do not duplicate them across hot-path instruction files.

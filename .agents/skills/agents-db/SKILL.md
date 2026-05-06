---
name: agents-db
description: Use when working with ARIA-NBV's internal agent memory (`.agents/AGENTS_INTERNAL_DB.md`, `.agents/issues.toml`, `.agents/todos.toml`, `.agents/refactors.toml`, `.agents/resolved.toml`) or triaging, resolving, and maintaining the backlog with `make agents-db`.
metadata:
  applies_to:
    - ".agents/AGENTS_INTERNAL_DB.md"
    - ".agents/issues.toml"
    - ".agents/todos.toml"
    - ".agents/refactors.toml"
    - ".agents/resolved.toml"
  triggers:
    - "agents DB"
    - "backlog"
    - "issue triage"
    - "resolved work"
  must_read:
    - ".agents/AGENTS_INTERNAL_DB.md"
    - ".agents/skills/agents-db/references/schema.md"
    - ".agents/skills/agents-db/references/provenance.md"
  verification:
    - "make agents-db AGENTS_ARGS='validate'"
    - "make agents-db"
    - "make check-agent-memory when memory or guidance changed"
---

# AGENTS DB

Use this skill when work depends on the internal agent-memory surfaces under
`.agents/`, active backlog ranking, or durable maintenance debt capture.

For behavior-preserving pruning or LOC reduction, also use `simplification`.

## Read First

1. `.agents/AGENTS_INTERNAL_DB.md`
2. `.agents/skills/agents-db/references/schema.md`
3. `.agents/skills/agents-db/references/provenance.md`
4. `.agents/skills/agents-db/references/modes.md` for `triage`,
   `to-issues`, or `to-prd` style work

## Workflow

1. Run or inspect `make agents-db` to understand active ranking.
2. Prefer amending existing records over creating duplicates.
3. Add or amend a record only when the work materially changes the repo's
   maintenance picture.
4. Keep records compact but auditable with `context` plus stable `references`.
5. Resolve or retire completed records into `.agents/resolved.toml`; do not
   delete records outright.

## Commands

- `make agents-db`
- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db AGENTS_ARGS='resolve issue issue-XXXX --note "..."'`
- `make agents-db AGENTS_ARGS='resolve todo todo-XXXX --note "..."'`
- `make agents-db AGENTS_ARGS='resolve refactor refactor-XXXX --note "..."'`

## Rules

- Use vertical slices for concrete follow-up work.
- Keep `.agents/*.toml` as the local source of truth unless the user explicitly
  asks to publish GitHub issues.
- Do not churn the DB for tiny local cleanup that does not change active debt.
- For broad or literature-backed additions, run a litkg route/query first and
  cite the relevant sources.

## Verification

- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `make check-agent-memory` when canonical memory, skills, debriefs, or guidance
  changed

# Agents DB Schema

Use this reference when adding or changing `.agents/issues.toml`,
`.agents/todos.toml`, `.agents/refactors.toml`, or `.agents/resolved.toml`.

## Record Owners

- `.agents/issues.toml`: active validated defects, integration gaps, and
  architectural debt.
- `.agents/todos.toml`: active concrete follow-up work linked to issue IDs.
- `.agents/refactors.toml`: active cleanup or simplification candidates.
- `.agents/resolved.toml`: resolved or intentionally retired issues, todos, and
  refactors.

## Required Fields

Issues must carry compact `context` and `references` so the claim can be
audited later.

Todos must define:

- `loc_min`, `loc_expected`, `loc_max`
- `issue_ids`
- `context`
- `references`
- `implementation_notes`
- `acceptance`
- `verification`

Refactors must define:

- `loc_min`, `loc_expected`, `loc_max`
- `issue_ids`
- `context`
- `implementation_notes`
- `acceptance`
- `verification`

Resolved records keep history instead of deleting prior context.

## Ranking

- Issues sort by priority, status, then ID.
- Todos sort by priority, status, lower `loc_expected`, then ID.
- Refactors sort by priority, status, lower `loc_expected`, then ID.

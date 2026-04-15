# Optional GitNexus Workflow

GitNexus is useful when it is available, but it is not a hard requirement for
the portable Codex scaffold.

## Use When Available
- Use GitNexus query/context/impact tools for unfamiliar code, refactors,
  renames, or impact analysis.
- Treat high or critical impact warnings as a reason to pause and explain the
  blast radius before editing.
- Prefer GitNexus rename tooling over text replacement for symbol renames.

## Fallback When Unavailable
- Use the nearest `AGENTS.md`, `scripts/nbv_get_context.sh match <term>`,
  targeted `rg`, and direct caller/callee inspection.
- Summarize the fallback impact analysis in the user-facing update when the edit
  affects public functions, classes, or package contracts.
- Run the targeted tests named by the nearest guide.

## Do Not
- Do not block routine Codex work solely because GitNexus MCP resources are not
  exposed in the current session.
- Do not leave root `AGENTS.md` with mandatory instructions that require
  unavailable tools.

# Human Owner Intent

Use this file for durable Jan-specific preferences that are not public project
narrative, not current technical truth, and not a repeatable workflow.

## Current Preferences

- Keep `.agents/` as the canonical agent scaffold.
- Keep OMX optional; it is orchestration, not repo state.
- Do not restore legacy cache migration or runtime training APIs.
- Keep retained QMD docs renderable, but separate current thesis, past seminar,
  and archived scratch material through folders and frontmatter.
- Manage checkpoints and model artifacts through Git LFS when they are intended
  to be versioned.
- Prefer compact ARIA-native skills over vendoring generic upstream skill sets.

## Instruction Capture

| New durable information | Destination |
|---|---|
| Repo-wide invariant or safety rule | `AGENTS.md` or nearest nested `AGENTS.md` |
| Repeatable workflow | `.agents/skills/<name>/SKILL.md` |
| Human-owner preference | `.agents/references/human_owner_intent.md` |
| Current project truth | `.agents/memory/state/*.md` |
| Actionable defect, todo, or refactor | `.agents/issues.toml`, `.agents/todos.toml`, or `.agents/refactors.toml` |
| Public thesis narrative | `docs/` Quarto or `docs/typst/seminar_paper/` |
| Generated routing/context | `docs/_generated/context/` or `.agents/kg/generated/` with provenance |

Prefer the smallest surface that can preserve the instruction without creating
a second source of truth.

# ARIA-NBV Source Order

Use this reference when a task needs current project truth or when sources
appear to disagree.

## Role Split

- Implemented substrate: `docs/typst/seminar_paper/main.typ` and included
  seminar-paper sections describe the system already written up and evaluated.
- Current thesis direction: `docs/contents/thesis/roadmap.qmd`,
  `docs/contents/thesis/questions.qmd`, and `.agents/memory/state/` describe
  the active thesis plan, locked decisions, open questions, gotchas, and current
  state.
- Advisor proposal narrative: `docs/typst/thesis/proposal.typ` and its included
  sections own proposal-facing wording once proposal work is in scope.
- Active maintenance work: `.agents/issues.toml`, `.agents/todos.toml`,
  `.agents/refactors.toml`, and `.agents/resolved.toml` via `make agents-db`.
- Generated routing artifacts: `docs/_generated/context/source_index.md`,
  `literature_index.md`, and `data_contracts.md`; refresh with `make context`
  when stale.
- Operator aids and long conventions: `.agents/references/`.

## Conflict Rule

When a source about future thesis scope conflicts with the seminar paper, keep
the seminar paper authoritative for implemented evidence and use current thesis
docs plus canonical memory for planned or locked future direction. Do not
silently promote planned work to implemented results.

## Capture Rule

- Repo invariant: root or nearest nested `AGENTS.md`.
- Repeatable workflow: `.agents/skills/*/SKILL.md`.
- Current truth: `.agents/memory/state/`.
- Actionable work: `.agents/issues.toml`, `.agents/todos.toml`, or
  `.agents/refactors.toml` through `agents-db`.
- Public narrative: Quarto or Typst docs.
- Human-owner preference: `.agents/references/human_owner_intent.md`.

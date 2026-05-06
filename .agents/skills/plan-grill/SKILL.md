---
name: plan-grill
description: Stress-test ARIA-NBV plans, thesis-scope decisions, advisor questions, and cross-surface refactors before implementation. Use when the request is vague, high-impact, research-facing, or needs one decision at a time against the paper, glossary, roadmap, memory state, and package guides.
metadata:
  applies_to:
    - "**"
  triggers:
    - "plan"
    - "advisor"
    - "scope decision"
    - "thesis"
    - "high-impact refactor"
    - "KG architecture"
  must_read:
    - ".agents/references/source_order.md"
    - "docs/contents/thesis/roadmap.qmd"
    - "docs/contents/thesis/questions.qmd"
    - ".agents/memory/state/PROJECT_STATE.md"
  verification:
    - "decision-complete plan with assumptions and deferred decisions"
---

# Plan Grill

## When To Use

Use this skill before implementing ambiguous or high-impact work:

- thesis scope, advisor-facing claims, or roadmap changes
- entity-aware RRI, target-conditioned VIN, rollout/RL, or simulator choices
- docs/public-internal partitioning, scaffold changes, or KG architecture
- broad cleanup where ownership, compatibility, or evidence bar is unclear

Do not use it for an already-localized bug with a concrete failing command; use
`diagnose-aria` instead.

## Grounding

Before asking the user, answer discoverable questions from:

1. `.agents/references/source_order.md`
2. `docs/typst/seminar_paper/main.typ` for implemented substrate claims
3. `.agents/memory/state/PROJECT_STATE.md`
4. `.agents/memory/state/DECISIONS.md`
5. `.agents/memory/state/OPEN_QUESTIONS.md`
6. `docs/typst/shared/glossary.typ`
7. `docs/contents/thesis/roadmap.qmd`
8. `docs/contents/thesis/questions.qmd`
9. the nearest `AGENTS.md` for touched code or docs

Use `aria-nbv-context` if the relevant surface is not yet localized.

## Interview Rules

- Ask one material decision at a time.
- State the recommended answer with the tradeoff.
- Challenge overloaded terms against `docs/typst/shared/glossary.typ`.
- For fuzzy thesis or planning terms, test the plan with three concrete
  scenarios: one normal case, one boundary case, and one failure case.
- Cross-check claims against code, paper, memory state, and roadmap before
  accepting them.
- Resolved terminology updates `docs/typst/shared/glossary.typ` or
  `.agents/memory/state/DECISIONS.md`. Do not add a parallel root context file
  or ADR tree as a second source of truth.
- Distinguish `current`, `planned`, `scratch`, and `archive` docs.
- Capture durable outcomes in the smallest correct surface:
  - invariant -> `AGENTS.md`
  - workflow -> `.agents/skills/`
  - current truth -> `.agents/memory/state/`
  - backlog -> `.agents/*.toml`
  - public narrative -> `docs/` or Typst
  - human preference -> `.agents/references/human_owner_intent.md`

## Output

End with a decision-complete plan that names:

- goal and success criteria
- in/out of scope
- public interfaces or docs surfaces affected
- implementation packages
- verification commands
- assumptions and deferred decisions

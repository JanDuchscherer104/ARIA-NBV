---
name: plan-grill
description: Stress-test vague, high-impact, research-facing, advisor-facing, or cross-surface ARIA-NBV decisions before implementation.
metadata:
  applies_to:
    - "**"
  triggers:
    - "advisor-facing decision"
    - "thesis scope"
    - "high-impact refactor"
    - "scaffold ownership"
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

Before asking the user, resolve discoverable facts from
`.agents/references/source_order.md` and the owning source for the decision.
Use `docs/typst/shared/glossary.typ` for overloaded terms and the nearest
`AGENTS.md` for touched code or docs.

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

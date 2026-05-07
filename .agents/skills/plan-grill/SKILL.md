---
name: plan-grill
description: Stress-test vague, high-impact, research-facing, advisor-facing, or cross-surface ARIA-NBV decisions before implementation.
metadata:
  mode: router
  not_when:
    - "a concrete failing command, traceback, or metric owns the task"
    - "the edit is already localized and low impact"
    - "the user asks for code review of concrete diffs"
  handoff_to:
    - "diagnose-aria for concrete failures"
    - "aria-nbv-context when the affected surface is unknown"
    - "code-review for concrete diff review"
    - "docs-curator for public narrative edits after the decision"
  evidence_required:
    - "source-order owner for the decision"
    - "success criteria, in/out of scope, and deferred decisions"
    - "claim-check output for advisor-facing claims"
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

## Grounding

Before asking the user, resolve discoverable facts from
`.agents/references/source_order.md` and the owning source for the decision.
Use `docs/typst/shared/glossary.typ` for overloaded terms and the nearest
`AGENTS.md` for touched code or docs.

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
- Capture durable outcomes through the root `AGENTS.md` Instruction Capture
  lanes.

## Output

End with a decision-complete plan that names:

- goal and success criteria
- in/out of scope
- public interfaces or docs surfaces affected
- implementation packages
- verification commands
- assumptions and deferred decisions

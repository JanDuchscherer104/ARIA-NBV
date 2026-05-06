---
name: agent-behavior
description: Use for all non-trivial ARIA-NBV coding, docs, scaffold, research, or memory edits before implementation to surface assumptions, minimize scope, make surgical changes, and define verification.
metadata:
  applies_to:
    - "**"
  triggers:
    - "non-trivial implementation"
    - "scaffold cleanup"
    - "docs or memory edits"
    - "research workflow changes"
  must_read:
    - "AGENTS.md"
    - ".agents/references/source_order.md"
  verification:
    - "surface-specific checks from .agents/references/verification_matrix.md"
    - "make check-agent-memory when agent guidance or memory changes"
---

# Agent Behavior

Apply this skill before non-trivial ARIA-NBV work. For obvious one-line fixes,
use judgment and keep the same principles lightweight.

## Principles

1. State assumptions and ambiguity before editing.
2. Prefer the simplest sufficient change.
3. Touch only task-relevant files and clean up only consequences of your own
   change.
4. Define success criteria and verification before declaring the work complete.
5. Match the local style and nearest ownership guide.

## ARIA Stop Rules

- Do not claim main target-conditioned performance from V0-only GT-input runs.
- Do not treat invalid candidates or unsupported targets as low-RRI valid
  samples.
- Do not route thesis-core planning through Gymnasium, SB3, Habitat, Isaac, or
  online RL until the M6 simulator bridge gate is explicitly in scope.
- Do not publish internal agent scaffolding, generated context, raw backlog, or
  OMX runtime material as public Quarto narrative.
- Do not create parallel truth surfaces; use the capture rules in `AGENTS.md`.

## Workflow

1. Localize the surface through root `AGENTS.md`, the nearest nested
   `AGENTS.md`, and the relevant skill metadata.
2. Name the intended behavior and success criteria in one or two sentences.
3. Choose the narrowest edit set that satisfies the criteria.
4. Run the verification for the touched surface.
5. Record durable truth, backlog, or debrief updates only when the work changes
   the maintenance or research state.

## Completion

- Every changed file maps to the user request or required verification.
- Any unverified item is called out explicitly.
- Any new durable rule, workflow, truth, preference, or action item is captured
  in the smallest correct surface named by root `AGENTS.md`.

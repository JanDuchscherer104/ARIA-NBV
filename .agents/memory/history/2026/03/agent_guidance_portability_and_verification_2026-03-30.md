---
id: 2026-03-30_agent_guidance_portability_and_verification
date: 2026-03-30
title: "Refine agent guidance portability, verification routing, and progressive disclosure"
status: done
topics: [codex, agents, workflow, memory]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - AGENTS.md
  - aria_nbv/AGENTS.md
  - .agents/references/operator_quick_reference.md
  - .agents/memory/state/DECISIONS.md
---

## Task

Resolve the shared-guidance portability issue, replace the global completion
checklist with a surface-specific verification matrix, and improve how the repo
routes agents into deeper guides.

## Method

Reviewed the root and nested `AGENTS.md` files, the operator quick reference,
and canonical workflow state. Rewrote the repo-root guidance to use portable
environment recovery commands, added dedicated progressive-disclosure routing at
the repo and package levels, and replaced the global "Done Means" block with a
verification matrix keyed to the touched surface. Updated the operator
reference to remove a stale `--extra pytorch3d` recovery command and recorded
the new workflow rules in canonical decisions.

## Findings

- The root guidance still embedded a host-specific `UV_PYTHON` path.
- The operator quick reference also contained a stale recovery command that no
  longer matched the declared extras in `aria_nbv/pyproject.toml`.
- Progressive disclosure already existed in nested guides, but the root file
  did not clearly tell agents when to descend into each boundary guide.

## Verification

- `make check-agent-memory`

## Canonical State Impact

Updated `.agents/memory/state/DECISIONS.md` to capture the portable shared
guidance rule, the localized progressive-disclosure flow, and the
surface-specific verification policy.

---
id: 2026-03-25_scaffold_operator_aids
date: 2026-03-25
title: "Restore operator aids without widening Codex bootstrap"
status: done
topics: [codex, scaffold, references]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - path: AGENTS.md
    kind: scaffold
  - path: .agents/references/operator_quick_reference.md
    kind: reference
  - path: .agents/skills/aria-nbv-context/SKILL.md
    kind: skill
  - path: .agents/skills/aria-nbv-context/scripts/nbv_context_index.sh
    kind: script
---

# Debrief

## Task
Tightened the Codex hot path, restored operator aids from the archived monolithic scaffold, and improved document-family discoverability without reintroducing broad startup context.

## Method
Compared the archived monolithic `AGENTS.md` against the current root scaffold, skill, and canonical state docs. Reintroduced only the high-value operational content as a small reference doc, then updated the source-index generator to expose curated documentation families.

## Verification
Regenerated lightweight context after the script changes and ran scaffold verification checks.

## Canonical State Impact
Updated `DECISIONS.md` and `PROJECT_STATE.md` to record the tightened bootstrap and the role of `.agents/references/`.

## Prompt Follow-Through

This note predates the privileged owner-directive memory contract. No additional durable owner prompt items were backfilled here beyond any canonical state updates already recorded in this debrief.

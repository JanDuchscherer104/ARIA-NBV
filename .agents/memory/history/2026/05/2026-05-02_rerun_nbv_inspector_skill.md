---
id: 2026-05-02_rerun_nbv_inspector_skill
date: 2026-05-02
title: "Rerun NBV Inspector Skill"
status: completed
topics: [skills, rerun, offline-store, diagnostics, geometry]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/skills/rerun-nbv-inspector/SKILL.md
  - .agents/skills/rerun-nbv-inspector/agents/openai.yaml
  - .agents/skills/rerun-nbv-inspector/references/context7-queries.md
  - .agents/skills/rerun-nbv-inspector/references/nbv-inspector-contract.md
  - .agents/skills/rerun-nbv-inspector/references/official-examples-map.md
  - .agents/skills/rerun-nbv-inspector/references/rerun-python-patterns.md
artifacts: []
---

# Rerun NBV Inspector Skill

## Task

Create a repo-local ARIA-NBV Rerun skill from the PRML/VSLAM
`rerun-slam-integration` skill, adapted to the ARIA-NBV offline inspector,
immutable VIN store, candidate/RRI ordering, PoseTW/CameraTW, PyTorch3D, and
display-only CW90 contracts.

## Method

Used the system `skill-creator` workflow and initialized
`.agents/skills/rerun-nbv-inspector` with references and UI metadata. Read the
PRML Rerun skill and its Rerun Python pattern, Context7 query, and official
example references. Queried current Rerun docs through Context7 for Python
recording setup, Pinhole, DepthImage, Transform3D, ViewCoordinates, LineStrips3D,
Points3D, Mesh3D, and RGB-D example patterns.

## Outcome

Added a compact `SKILL.md` with a workflow, guardrails, review checklist, and
verification commands for ARIA-NBV Rerun work. Added reference files for
Context7 queries, Rerun Python patterns, the ARIA-NBV inspector contract, and
official example selection. Fixed the generated `agents/openai.yaml` prompt so
it preserves `$rerun-nbv-inspector`.

## Verification

`aria_nbv/.venv/bin/python /home/jd/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/rerun-nbv-inspector` passed. Searched the new skill for leftover template TODO text and found none.

## Canonical Impact

No canonical state update is needed. The skill is the durable repeatable-workflow
surface for future ARIA-NBV Rerun inspector creation, review, and repair tasks.

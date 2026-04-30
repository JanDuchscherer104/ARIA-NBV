---
id: 2026-04-30_lrz_ai_systems_skill
date: 2026-04-30
title: "LRZ AI Systems Skill"
status: done
topics: [skills, lrz, slurm, dss, containers]
confidence: high
canonical_updates_needed: []
files_touched:
  - AGENTS.md
  - .agents/skills/lrz-ai-systems/SKILL.md
  - .agents/skills/lrz-ai-systems/references/
  - .agents/skills/lrz-ai-systems/scripts/
  - .agents/skills/lrz-ai-systems/templates/
---

## Task

Created an ARIA-NBV repo skill for operating safely on LRZ AI Systems as a shell/operator workflow only.

## Method

Added compact skill routing, DSS storage guidance, Slurm resource helpers, Pyxis container guidance, sbatch wrappers, editable templates, and a root routing bullet. Kept Python runtime APIs untouched.

## Verification

Planned local checks: shell syntax, executable bits, secret/path scan, skill validation, agent-memory validation, and agents-db validation.

## Canonical State Impact

No project state files need updates beyond the new skill and root routing bullet.

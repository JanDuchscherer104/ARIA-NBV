---
id: 2026-04-16_workflow_skills_adaptation
date: 2026-04-16
title: "Adapt imported GitHub/review workflow skills to ARIA-NBV"
status: done
topics: [skills, github, review, issues, pr]
confidence: high
canonical_updates_needed: []
files_touched:
  - path: .agents/skills/code-review/SKILL.md
    kind: skill
  - path: .agents/skills/create-gh-pr/SKILL.md
    kind: skill
  - path: .agents/skills/gh-issue-lifecycle/SKILL.md
    kind: skill
  - path: .github/ISSUE_TEMPLATE/01-backlog-item.yml
    kind: template
---

## Task

Adapt imported review, PR, and GitHub issue lifecycle skills so they reference
real ARIA-NBV repo surfaces and commands.

## Verification

- `/home/jandu/repos/NBV/aria_nbv/.venv/bin/python /home/jandu/repos/NBV-packages/workflow-skills-adaptation/scripts/validate_agent_scaffold.py`

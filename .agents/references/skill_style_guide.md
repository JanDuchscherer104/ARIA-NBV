# ARIA-NBV Skill Style Guide

Skills are hot-path agent instructions. Keep them compact, task-specific, and
activation-oriented. Long details belong in `references/` files.

## Required Frontmatter

```yaml
---
name: skill-name
description: One sentence describing when to use the skill.
metadata:
  applies_to:
    - "repo/glob/**"
  triggers:
    - "phrase or task cue"
  must_read:
    - "small required source list"
  verification:
    - "command or review check"
---
```

Use meaningful routing metadata under `metadata:` so skills stay compatible with
the local skill validator. Broad skills may use broad globs, but triggers must
still be concrete enough for an agent or KG router to distinguish them.

## Body Template

- Use When
- Do Not Use When, if confusion is likely
- Read First, usually 3-5 sources
- Rules, usually 5-10 bullets
- Workflow, short and ordered
- Verification
- Stop or completion conditions

## Style Rules

- Default skill bodies should stay under about 150 lines unless the skill wraps
  an operator workflow with unavoidable commands.
- Avoid duplicating root source order, long command lists, or schema manuals.
- Prefer references over nested procedural walls.
- Do not add speculative abstractions or future-work instructions unless the
  task explicitly owns that future-work surface.
- Every skill should preserve the `agent-behavior` principles: explicit
  assumptions, simplest sufficient change, surgical edits, and verifiable
  completion.

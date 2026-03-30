---
id: 2026-03-29_deep_research_scaffold_gap_closure
date: 2026-03-29
title: "Deep-Research Scaffold Gap Closure"
status: done
topics: [scaffold, codex, metadata, docs]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - AGENTS.md
  - aria_nbv/AGENTS.md
  - docs/AGENTS.md
  - .agents/memory/state/DECISIONS.md
  - scripts/validate_agent_memory.py
---

# Task

Closed the remaining scaffold gaps from the deep-research report by adding YAML frontmatter to repo-managed `AGENTS.md` files, clarifying the root purpose/setup intro, and adding a short maintainer checklist without expanding the hot path into workflow or security policy.

# Method

- Added small YAML frontmatter blocks to the repo-root, package, and docs `AGENTS.md` files.
- Reworked the root scaffold opening into `Purpose` plus `Setup & Bootstrap`, grounded in the paper narrative and current ideas-driven direction.
- Pointed setup guidance to `docs/contents/setup.qmd` instead of duplicating environment instructions in the root scaffold.
- Added a short root `Maintenance Checklist`.
- Extended scaffold validation so repo-managed `AGENTS.md` files must carry valid frontmatter keys.

# Verification

- `make check-agent-scaffold`
- `make check-agent-memory`

# Canonical State

- Updated `.agents/memory/state/DECISIONS.md` to record YAML frontmatter on repo-managed `AGENTS.md` files and the canonical setup pointer to `docs/contents/setup.qmd`.

## Prompt Follow-Through

- Applied the chosen deep-research report recommendations: stronger root purpose/setup framing, YAML frontmatter on all repo-managed `AGENTS.md` files, and a compact root maintainer checklist.
- Left out workflow, security, extra skill metadata, and behavioral harness additions as requested.
- No additional durable owner prompt items needed promotion beyond the implemented scaffold choices and the `DECISIONS.md` update.

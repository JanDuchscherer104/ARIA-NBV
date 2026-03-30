---
id: 2026-03-29_root_package_guidance_split
date: 2026-03-29
title: "Root vs Package Guidance Split"
status: done
topics: [scaffold, codex, memory, package-guidance]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
files_touched:
  - AGENTS.md
  - aria_nbv/AGENTS.md
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/PROJECT_STATE.md
---

# Task

Moved the remaining Python, implementation, and package-verification guidance out of the repo-root scaffold and into `aria_nbv/AGENTS.md`.

# Method

- Trimmed the root `AGENTS.md` to repo-wide bootstrap, scaffold commands, shared scope constraints, and generic validation routing.
- Expanded `aria_nbv/AGENTS.md` to carry the package environment recovery command, package contract lookup command, `PoseTW` / `CameraTW`, `.setup_target()`, and package verification rules.
- Updated canonical state docs so the root-vs-package split is recorded as current scaffold policy.

# Verification

- `make check-agent-memory`

# Canonical State

- Updated `.agents/memory/state/DECISIONS.md` to state that package implementation guidance lives in `aria_nbv/AGENTS.md`.
- Updated `.agents/memory/state/PROJECT_STATE.md` to reflect the repo-wide vs path-local `AGENTS.md` split.

## Prompt Follow-Through

This note predates the privileged owner-directive memory contract. No additional durable owner prompt items were backfilled here beyond any canonical state updates already recorded in this debrief.

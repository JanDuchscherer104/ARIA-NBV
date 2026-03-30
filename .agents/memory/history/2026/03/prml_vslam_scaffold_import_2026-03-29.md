---
id: 2026-03-29_prml_vslam_scaffold_import
date: 2026-03-29
title: "Import High-Signal Scaffold Ideas From prml-vslam"
status: done
topics: [scaffold, codex, package-guidance, context7]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/OWNER_DIRECTIVES.md
files_touched:
  - AGENTS.md
  - aria_nbv/AGENTS.md
  - .agents/references/context7_library_ids.md
  - .agents/memory/state/OWNER_DIRECTIVES.md
---

# Task

Imported the high-signal scaffold ideas from the `prml-vslam` root and package-local `AGENTS.md` files into the NBV scaffold without copying their low-value commit or workflow chatter.

# Method

- Added a compact repo map to the repo-root `AGENTS.md` and renamed the root sections to be more coherent and intuitive.
- Reworked `aria_nbv/AGENTS.md` around clearer package-local groupings: core rules, config-as-factory, anti-patterns, code quality, verification, and completion criteria.
- Added explicit `BaseConfig` / `.setup_target()` guidance and anti-patterns to the package-local scaffold.
- Expanded the on-demand Context7 library index with verified entries for Lightning, W&B, and Optuna.
- Recorded the owner feedback about section titles, grouping coherence, and discoverability of project-specific patterns in `OWNER_DIRECTIVES.md`.

# Verification

- `make check-agent-scaffold`
- `make check-agent-memory`

# Canonical State

- Updated `.agents/memory/state/OWNER_DIRECTIVES.md` to preserve the owner feedback about coherent scaffold grouping and discoverable project-specific implementation patterns.

## Prompt Follow-Through

- Captured the owner feedback that some section titles and groupings in `AGENTS.md` files did not feel coherent or intuitive.
- Promoted that durable guidance into `.agents/memory/state/OWNER_DIRECTIVES.md` instead of leaving it only in chat.
- Applied the suggested `prml-vslam` imports at the appropriate layer: tiny repo map in the root scaffold, config-factory and anti-pattern guidance in `aria_nbv/AGENTS.md`, and curated external-library IDs in `.agents/references/context7_library_ids.md`.

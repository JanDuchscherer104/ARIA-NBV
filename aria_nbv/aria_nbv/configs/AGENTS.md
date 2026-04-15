---
scope: module
applies_to: aria_nbv/aria_nbv/configs/**
summary: Persistent config, path, W&B, and Optuna guidance for Aria-NBV.
---

# Config Guidance

Follow [../../AGENTS.md](../../AGENTS.md) plus this file for work under
`aria_nbv/aria_nbv/configs/`. Durable config ownership notes live in
[README.md](README.md).

## Rules
- Keep durable workflow configuration in typed `BaseConfig` models.
- Use `.setup_target()` for runtime construction and keep validation/default
  wiring inside config models when it is not runtime-only behavior.
- Keep host-specific paths out of shared defaults; put machine-local recovery
  details in operator references instead.
- Treat W&B, Optuna, and path config changes as workflow-facing behavior changes
  and update docs or memory when semantics change.

## Verification
- Run `ruff format` and `ruff check` on touched config files.
- Run targeted tests or CLI/config smoke tests when config loading, path
  resolution, sweeps, or experiment setup changes.

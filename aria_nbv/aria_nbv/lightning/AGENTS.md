---
scope: module
applies_to: aria_nbv/aria_nbv/lightning/**
summary: Training experiment, datamodule, module, trainer, and CLI guidance.
---

# Lightning Guidance

Follow [../../AGENTS.md](../../AGENTS.md) plus this file for work under
`aria_nbv/aria_nbv/lightning/`. Durable training ownership notes live in
[README.md](README.md).

## Rules
- Keep experiment construction config-driven through `.setup_target()`.
- Treat datamodule source selection, split handling, batch collation, and
  validation enablement as training-facing contracts.
- Validation is disabled unless the trainer config explicitly enables it; do not
  assume validation runs by default.
- Keep W&B, Optuna, callbacks, and trainer settings explicit and reproducible.

## Verification
- Run targeted Lightning tests when changing datamodule sources, batch collation,
  trainer factory behavior, validation settings, or CLI entrypoints.
- Run a CLI `--help` smoke test for changed operator-facing training commands.

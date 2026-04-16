---
scope: module
applies_to: aria_nbv/aria_nbv/pipelines/**
summary: Pipeline entrypoint and oracle-labeler orchestration guidance.
---

# Pipeline Guidance

Follow [../../AGENTS.md](../../AGENTS.md) plus this file for work under
`aria_nbv/aria_nbv/pipelines/`. Durable pipeline ownership notes live in
[README.md](README.md).

## Rules
- Keep pipeline modules as orchestration layers around typed package contracts.
- Treat oracle-labeler output shape, cache writes, and operator-facing CLI
  behavior as workflow contracts.
- Do not hide expensive or lossy fallback behavior inside pipelines; fail with
  actionable rebuild or configuration guidance.
- Keep narrative docs aligned when a pipeline changes user-visible workflow or
  artifact semantics.

## Verification
- Run targeted pipeline, data-handling, rendering, or RRI tests for changed
  pipeline behavior.
- Run `--help` smoke tests for changed pipeline CLIs or scripts.

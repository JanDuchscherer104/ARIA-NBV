---
id: 2026-05-06_setup_smoke_docs_worker_c
date: 2026-05-06
title: "Setup And One-Scene Smoke Docs"
status: done
topics: [setup, docs, offline-store, rerun, vin]
confidence: high
canonical_updates_needed:
  - .agents/todos.toml
files_touched:
  - README.md
  - SETUP.md
  - docs/contents/setup.qmd
  - docs/contents/impl/one_scene_smoke.qmd
---

## Task

Worker C implemented the portable setup and one-scene smoke documentation
surface while leaving unrelated proposal, bibliography, tests, Makefile/CI,
LRZ templates, and storage schema changes untouched.

## Method

Used the docs-curator workflow, read the public docs guide, paper entry point,
project state, gotchas, and a litkg context pack for the setup-doc task. Checked
the active CLI/config surfaces for `nbv-build-offline`, `nbv-rerun-inspect`,
`nbv-summary`, `nbv-rollout-trace-smoke`, `PathConfig`, and the smoke TOMLs
before writing commands into the docs.

## Outputs

Added root `SETUP.md`, refreshed `docs/contents/setup.qmd`, added
`docs/contents/impl/one_scene_smoke.qmd`, and added short README links. The docs
now cover portable environment setup, CUDA/CPU expectations, dataset/cache
paths, immutable VIN offline stores, Rerun inspection, VIN diagnostics, one
sample smoke commands, synthetic rollout trace smoke, and docs sanity commands.

## Verification

- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/setup.qmd`
- `cd docs && quarto render contents/impl/one_scene_smoke.qmd`
- `cd docs && quarto render contents/impl/rerun_offline_inspector.qmd`
- `cd docs && quarto check`
- `cd aria_nbv && uv run nbv-build-offline --config-path ../.configs/build_vin_offline_rerun_smoke_v6.toml --dry-run`
- `cd aria_nbv && uv run nbv-rollout-trace-smoke --output-path ../.artifacts/rollouts/docs_synthetic_rollout_traces.msgpack`
- `make check-agent-memory`

## Canonical State Impact

`todo-002` in the agents DB appears ready for backlog resolution after the
multi-worker worktree is integrated. It was not edited in this task to avoid
cross-worker ownership conflicts.

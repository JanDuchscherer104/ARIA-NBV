---
id: 2026-05-06_m0_m1_parallel_unblocker
date: 2026-05-06
title: "M0/M1 Parallel Unblocker Sprint"
status: done
topics: [m0, m1, ci, setup, oracle, rollout-storage, lrz, proposal]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/issues.toml
  - .agents/todos.toml
  - .agents/refactors.toml
  - .agents/resolved.toml
  - .github/workflows/ci.yml
  - .pre-commit-config.yaml
  - Makefile
  - README.md
  - SETUP.md
  - aria_nbv/aria_nbv/data_handling/_offline_writer.py
  - aria_nbv/aria_nbv/pose_generation/types.py
  - aria_nbv/aria_nbv/rendering/candidate_depth_renderer.py
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
  - aria_nbv/tests/pose_generation/test_counterfactuals.py
  - docs/contents/thesis/m1_contract_report.qmd
  - docs/contents/impl/one_scene_smoke.qmd
  - docs/contents/impl/rollout_storage_contract.qmd
  - docs/contents/impl/lrz_dry_runs.qmd
  - .agents/references/rollout_zarr_q_invalidity_contract.md
---

## Task

Implemented the M0/M1 parallel unblocker plan after a rebase/autostash cleanup.
The sprint covered advisor-facing proposal/docs polish, candidate-label
ordering guards, root CI/pre-commit checks, setup and one-scene smoke docs,
a draft rollout/Q_H storage contract, and dry-run LRZ Slurm/DSS templates.

## Method

The coordinator first inspected the dirty worktree and `stash@{0}`. The stash
contained no unique changes relative to the rebased tree and was dropped. A
small frontmatter-validator fix was committed separately as
`be219df Allow read-only archive scratch docs`. Independent workers then edited
disjoint docs, test, CI, storage-schema, and LRZ surfaces; the coordinator
integrated sidebar/overview links, backlog resolution, and M1 gate wording.

## Outputs

- Added synthetic candidate-order and candidate-label alignment guards around
  candidate shell indices, rendered depth rows, RRI vectors, camera tensors,
  optional point clouds, and offline sample payloads.
- Added `docs/contents/thesis/m1_contract_report.qmd` with explicit pass/block
  sections for source contracts, synthetic checks, real-store evidence, Rerun
  recordings, and oracle throughput.
- Added root `make ci`, local pre-commit parity hooks, and a minimal GitHub
  Actions root verification workflow.
- Added `SETUP.md`, refreshed public setup docs, and added a one-scene smoke
  workflow page.
- Added draft rollout/Zarr/Q_H/invalidity contracts without implementing large
  writers or starting rollout scale.
- Added documentary LRZ dry-run templates and DSS staging guidance.

## Verification

- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `make qmd-frontmatter-check`
- `make check-agent-memory`
- targeted candidate-order pytest command covering offline-store, pose
  generation, Lightning collate, and VIN shuffle surfaces
- `make proposal-pdf`
- `bash -n scripts/templates/lrz/*.sbatch`
- TOML parse check for `.configs/lrz/dry_run_matrix.toml`
- targeted Quarto renders for M1, setup, smoke, rollout storage, LRZ, roadmap,
  questions, and literature pages
- `make ci`
- `git diff --check -- ':!docs/typst/thesis/proposal.pdf'`

## Canonical State Impact

No additional canonical state update is required. The sprint preserves the
current gate: M1 remains blocked until real-store diagnostics, Rerun evidence,
and oracle throughput are collected. V0 target-RRI, stochastic rollout scale,
and fitted Q_H should not start before that gate is passable.

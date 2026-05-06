---
id: 2026-05-06_rollout_zarr_q_invalidity_schema
date: 2026-05-06
title: "Rollout Zarr Q Invalidity Schema"
status: done
topics: [rollouts, storage, invalidity, q-learning, docs]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/impl/rollout_storage_contract.qmd
  - docs/contents/impl/overview.qmd
  - docs/contents/thesis/roadmap.qmd
  - docs/contents/thesis/questions.qmd
  - docs/_quarto.yml
  - .agents/references/rollout_zarr_q_invalidity_contract.md
---

# Rollout Zarr Q Invalidity Schema

## Task

Drafted the rollout Zarr/Q_H/invalidity schema only, without implementing
writers, migrations, stochastic rollout generation, Q_H training, tests, setup,
CI, bibliography, or LRZ templates.

## Outputs

- Added a developer-facing Quarto implementation page for the rollout storage
  and invalidity contract.
- Added an internal `.agents/references` contract with mechanical field tables,
  row ids, masks, reason codes, external mesh references, target crop handling,
  optional diagnostics, and padded Q_H tensor shapes.
- Linked the new page from the implementation overview, docs navigation,
  roadmap M2/M5, and thesis RQ7 storage gate.

## Verification

Passed after edits:

- `make qmd-frontmatter-check`
- `make check-agent-memory`
- `cd docs && quarto render contents/impl/rollout_storage_contract.qmd`
- `cd docs && quarto render contents/impl/overview.qmd`
- `cd docs && quarto render contents/thesis/roadmap.qmd`
- `cd docs && quarto render contents/thesis/questions.qmd`
- `git diff --check`
- `scripts/nbv_qmd_outline.sh --compact | rg -n "rollout_storage_contract|Rollout Storage"`

One first combined Quarto command with multiple page arguments failed because
this Quarto/Pandoc invocation treated the extra page paths as Pandoc inputs.
The four touched pages rendered successfully when run individually.

## Canonical State Impact

No durable state update is pending. Existing project state already tracked
Zarr-first rollout/Q storage and invalidity masks/reasons as prerequisite
contract work.

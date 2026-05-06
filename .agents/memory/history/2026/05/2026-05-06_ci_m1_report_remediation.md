---
id: 2026-05-06_ci_m1_report_remediation
date: 2026-05-06
title: "CI and M1 Report Review Remediation"
status: done
topics: [ci, m1, docs, agents-db]
confidence: high
canonical_updates_needed: []
files_touched:
  - .github/workflows/ci.yml
  - Makefile
  - docs/contents/thesis/m1_contract_report.qmd
  - .agents/todos.toml
---

## Task

Implemented the remediation plan for four review findings against commit
`d6c2cba`: broaden root CI triggers, add a lightweight package smoke gate to
`make ci`, turn the M1 report into an explicit evidence ledger, and repair the
stale `todo-046` provenance link.

## Outputs

- Root CI now triggers on package, setup, and config/template surfaces that were
  part of the M0/M1 groundwork.
- `make ci` now includes `package-smoke`, which runs bounded Ruff format/lint
  checks and CPU-only package tests covering data handling, public API,
  candidate/rollout ordering, rendering, and VIN batch masking.
- `docs/contents/thesis/m1_contract_report.qmd` records the current local
  offline store evidence: manifest v6, manifest hash, 48 sample-index rows, one
  scene, 38/10 train/val sample split, interrupted/partial status, missing
  Rerun recordings, and missing throughput evidence.
- `todo-007` now points at `repo:.agents/resolved.toml#todo-046`.

## Verification

- `make package-smoke`
- `cd docs && quarto render contents/thesis/m1_contract_report.qmd`
- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `CI_RENDER_DIR=/tmp/aria-nbv-ci-renders-remediation make ci`

## Canonical State Impact

No canonical state update is needed. M1 remains blocked on real scene-level
split evidence, Rerun normal/boundary/failure recordings, and representative
oracle throughput evidence.

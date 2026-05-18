---
id: 2026-05-18_data_generation_runbook
date: 2026-05-18
title: "Data Generation Runbook"
status: done
topics: [docs, data, rollouts, lrz, cli]
confidence: high
canonical_updates_needed: []
files_touched:
  - .configs/lrz/README.md
  - aria_nbv/aria_nbv/data_handling/README.md
  - docs/contents/setup.qmd
---

## Task

Integrated the ARIA-NBV data-generation CLI workflow into the package
data-handling README so the runbook is visible to both agents and human
operators.

## Method

Made `aria_nbv/aria_nbv/data_handling/README.md` the canonical surface for both
storage contracts and the operational workflow: raw ASE/ATEK-EFM download, VIN
offline-store generation, rollout smoke generation, rollout shard
planning/build/status, LRZ campaign execution, and troubleshooting. Updated the
public setup page and LRZ DSS README to point at that package README instead of
an agent-only reference file.

## Verification

- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/setup.qmd`
- `make check-agent-memory`
- `git diff --check`

## Canonical State Impact

No canonical thesis-state update is needed. The change records an operator
workflow location and does not change data contracts or thesis direction.

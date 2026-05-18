---
id: 2026-05-18_rollout_review_triage_agents_db
date: 2026-05-18
title: "Rollout Review Triage Agents DB"
status: done
topics: [agents-db, rollouts, data-generation, lrz, thesis]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/issues.toml
  - .agents/todos.toml
---

## Task

Persist the accepted rollout dataset review findings from
`.agents/work/rolllout_data/rollout-review-01-gpt55pro.md` and
`.agents/work/rolllout_data/rollout-review-02-gpt55pro.md` into existing
agents DB records only.

## Method

Compared the reviews against current code, local store manifests, rollout docs,
and active DB records. Amended existing issues and todos rather than creating
duplicate records or new dataset surfaces.

## Outputs

- Clarified that `vin_offline` is immutable root/source substrate; the current
  local store is a partial one-scene strict-v7 seminar cache, not
  target-conditioned thesis evidence.
- Recorded H=1 rollout sidecar/profile as the one-step target-conditioned label
  gate before H3+ multi-step rollout scale, without a new
  `TargetLabelDatasetWriter`.
- Corrected stale rollout smoke status: `rollouts_v1_smoke.zarr` now validates
  as schema `0.5-selected-depth`, but remains an H3/N5 one-scene smoke artifact.
- Added LRZ preflight, scene-level split, target-threshold/crop audit,
  invalid-reason population, stochastic RNG replay, Q_H selected-transition TD,
  and schema-pruning notes to the existing backlog.

## Verification

- `make agents-db AGENTS_ARGS='validate'` passed after TOML edits.

## Canonical State Impact

No canonical state updates were needed; this pass updated active backlog records
only.
